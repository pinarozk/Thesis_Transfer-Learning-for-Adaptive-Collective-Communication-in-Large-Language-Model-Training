"""
Agent training script (thesis-ready).

Fixes vs. previous version
--------------------------
1. NO SILENT REPAIRS: standardize_feature_dims (zero-padding / truncating
   features) and edge_align_target (matrix->vector conversion) were
   train-time patches for schema inconsistencies that the new pipeline
   fixes at the source. Padding features with zeros silently changes
   feature SEMANTICS across samples — unacceptable. Both are replaced
   with hard assertions: if the schema is inconsistent, generation is
   broken and must be fixed there.
2. [REMOVED] pos_weight / BCE class-imbalance handling: gone along with
   the binary routing head (fix #8 below) — a regression target has no
   class imbalance to correct for.
3. LOSS CONTRACT FROM THE MODEL: model.compute_losses() is used, so the
   masked scheduling loss (sched_mask) cannot silently diverge between
   training and evaluation code paths.
4. MULTI-SEED: results in the thesis must carry mean ± std over seeds,
   not a single lucky run. SEEDS list; per-seed checkpoints; aggregated
   summary saved as JSON.
5. EVALUATION THAT MATTERS:
     - load: MAE / RMSE / Pearson & Spearman correlation (log1p scale),
       plus a diagnostic used-edge precision/recall/F1 from
       thresholding load_pred (NOT the primary metric — see fix #8)
     - scheduling: masked MAE / RMSE
     - PER-TOPOLOGY breakdown on test (group_key) — with group-based
       splits, aggregate test numbers hide which structures transfer.
6. [REMOVED] post-hoc BCE temperature calibration: no probability to
   calibrate once routing is a regression target (see fix #8).
7. Reproducibility: full config + git-friendly JSON results, seeds fixed
   for torch / python; cudnn determinism enabled.
8. ROUTING -> LOAD REGRESSION: real data showed routing_positive_ratio
   ~0.99-1.00 across all three teachers on these topologies (AllGather
   touches nearly every edge at least once on small/dense graphs) --
   the binary "used or not" target was near-degenerate (a trivial
   always-positive baseline scores ~1.0 F1). Per-edge chunk load
   (log1p-scaled count/amount) is the informative target: it is what
   the model was already implicitly using (the capacity-sensitivity
   probe showed 570x response even under the old, blind binary head),
   and it is what determines makespan (bottleneck edge = highest-load
   edge).

Pairs with: agent_model.py, convert_to_pyg.py outputs.
"""

import sys
import json
import random
from pathlib import Path
from collections import defaultdict

import torch
from torch_geometric.loader import DataLoader

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.models.agent_gnn import StrongAgentGNN  # noqa: E402


# ============================================================
# Paths
# ============================================================

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "pyg_agent_train.pt"
VAL_PATH = DATA_DIR / "pyg_agent_val.pt"
TEST_PATH = DATA_DIR / "pyg_agent_test.pt"
STATS_PATH = DATA_DIR / "pyg_agent_stats.pt"

RESULTS_PATH = MODEL_DIR / "agent_results.json"


# ============================================================
# Config
# ============================================================

SEEDS = [42, 1337, 2024]        # fix #4: report mean ± std

BATCH_SIZE = 32
NUM_EPOCHS = 200
PATIENCE = 25

LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 2.0

HIDDEN_DIM = 64                 # conservative until dataset grows
HEADS = 4
NUM_LAYERS = 3
DROPOUT = 0.15

LAMBDA_SCHED = 0.5
SCHEDULING_ACTIVATION = "sigmoid"   # target lives in (0, 1]

# Switch-edge upweighting in the load loss (see agent_gnn.compute_losses
# load_edge_weight). Added after cct_gap.py showed switch-heavy held-out
# topologies (DGX2_2_chassis) cost 33x downstream vs 2.7x switchless,
# traced to switch edges being <10% of train edges (see EDGE_FEATURE_NAMES
# index below). SWITCH_UPWEIGHT=1.0 reproduces the unweighted baseline.
SWITCH_UPWEIGHT = 1.0  # tried 5.0: no downstream CCT-gap improvement
                        # (33.12x -> 33.93x on switch topologies) --
                        # confirmed topology-family generalization gap,
                        # not a class-imbalance problem. Left at 1.0
                        # (no-op) rather than removed, so the mechanism
                        # stays available/documented for future work.
SWITCH_EDGE_FEAT_NAME = "is_switch_edge"

# "residual": train the load head against y_load_residual (= y_load -
# sp_load_log1p, the analytic shortest-path-load prior; see
# load_priors.py) instead of raw y_load. Motivation: the model is
# blind on switch edges of unseen topology families (r=0.03 on the
# DGX2_2_chassis holdout -> 33x downstream CCT gap), and switch-edge
# upweighting proved this is not a class-imbalance problem -- what's
# missing is INFORMATION about an unseen family's load regime. The
# analytic prior alone reaches r=0.475 on that same holdout edges,
# and (unlike a learned pattern) is a computation that transfers to
# any topology. Training on the residual lets the model correct that
# baseline rather than relearn it from scratch; absolute load is
# always recoverable as pred_residual + sp_load_log1p, so an OOD
# failure degrades to "baseline + noise" instead of a blind guess.
# "absolute" reproduces the pre-residual baseline exactly.
#
# TRIED residual, REVERTED: edge-level correlation improved (Pearson
# 0.603 -> 0.690, Spearman -> 0.697) but downstream CCT-gap got WORSE
# across the board (overall 24.71x -> 48.41x, switch 33.12x -> 65.41x,
# switchless 2.69x -> 3.87x). Root-cause hypothesis: sp_load_log1p is a
# pure shortest-path prior, but AllGather's real optimum deliberately
# AVOIDS shortest-path concentration to spread load across parallel
# trees (that's the entire reason a solver is needed instead of
# routing everything via Dijkstra). Dominating the load head with this
# prior pushes decode_allgather's per-source Dijkstra tree toward a
# more shortest-path-like, more bottlenecked schedule -- better
# pointwise correlation with true load, worse structural decode.
# Second confirmed instance (after SWITCH_UPWEIGHT) that improving the
# edge-level proxy metric does not transfer to -- and here actively
# hurts -- the downstream metric that actually matters.
LOAD_TARGET = "absolute"

# Diagnostic-only cut for the "is this edge used" side metric, in
# log1p(load) space. log1p(1) == the model predicting >=1 whole chunk
# crossed the edge. NOT used by the loss (see agent_gnn.compute_losses).
LOAD_USED_THRESHOLD = 0.6931471805599453  # math.log1p(1.0)

REQUIRED_KEYS = {
    "x", "edge_index", "edge_attr",
    "y_load", "y_load_residual", "sp_load_log1p",
    "y_routing", "y_scheduling", "sched_mask", "u",
}


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Data loading + contract validation (fix #1)
# ============================================================

def load_split(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{name}: expected non-empty list at {path}")
    return data


def validate_schema(datasets):
    """
    Hard contract: identical feature dims everywhere, edge-aligned
    targets, mask coherence. Any violation is an upstream bug —
    raise, never pad/truncate/convert here.
    """
    node_dims, edge_dims, u_dims = set(), set(), set()

    for split in datasets:
        for d in split:
            node_dims.add(d.x.size(1))
            edge_dims.add(d.edge_attr.size(1))
            u_dims.add(d.u.size(-1) if d.u.dim() > 1 else d.u.size(0))

            ne = d.edge_index.size(1)
            assert d.y_load.dim() == 1 and d.y_load.numel() == ne, \
                f"y_load not edge-aligned in {d.topology_name}"
            assert d.y_load_residual.numel() == ne
            assert d.sp_load_log1p.numel() == ne
            assert torch.allclose(
                d.y_load - d.sp_load_log1p, d.y_load_residual, atol=1e-4
            ), f"residual != load - baseline in {d.topology_name}"
            assert d.y_scheduling.numel() == ne
            assert d.sched_mask.numel() == ne
            assert torch.equal(d.sched_mask.bool(), d.y_load.bool())

    if len(node_dims) != 1 or len(edge_dims) != 1 or len(u_dims) != 1:
        raise ValueError(
            f"Inconsistent feature dims across dataset — fix the "
            f"pipeline, do NOT pad here. node={node_dims}, "
            f"edge={edge_dims}, u={u_dims}"
        )

    return node_dims.pop(), edge_dims.pop(), u_dims.pop()


def detect_exclude_keys(*datasets):
    all_keys = set()
    for split in datasets:
        for d in split[:50]:
            all_keys.update(d.keys())
    return sorted(all_keys - REQUIRED_KEYS)


def prepare_u(batch):
    u = getattr(batch, "u", None)
    if u is None:
        return None
    num_graphs = int(batch.batch.max().item()) + 1 \
        if batch.batch is not None and batch.batch.numel() else 1
    if u.dim() == 1:
        if num_graphs > 1 and u.numel() % num_graphs == 0:
            u = u.view(num_graphs, -1)
        else:
            u = u.unsqueeze(0)
    return u


# ============================================================
# Metrics (dependency-free implementations)
# ============================================================

@torch.no_grad()
def binary_metrics(preds01, targets01):
    """preds01/targets01 already thresholded to {0,1} -- used only for
    the diagnostic used-edge side metric, not the primary load metric."""
    tp = float(((preds01 == 1) & (targets01 == 1)).sum())
    fp = float(((preds01 == 1) & (targets01 == 0)).sum())
    fn = float(((preds01 == 0) & (targets01 == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


@torch.no_grad()
def pearson_corr(a, b):
    a, b = a.float(), b.float()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp(min=1e-12)
    return float((a * b).sum() / denom)


@torch.no_grad()
def spearman_corr(a, b):
    """Pearson correlation of ranks -- monotonic-relationship signal,
    robust to the heavy-tailed load distribution's outliers."""
    ra = torch.argsort(torch.argsort(a)).float()
    rb = torch.argsort(torch.argsort(b)).float()
    return pearson_corr(ra, rb)


@torch.no_grad()
def regression_metrics(pred, target):
    err = pred - target
    return {
        "mae": float(err.abs().mean()),
        "rmse": float((err ** 2).mean().sqrt()),
        "pearson_r": pearson_corr(pred, target),
        "spearman_r": spearman_corr(pred, target),
    }


# ============================================================
# Epoch loops
# ============================================================

def run_epoch(model, loader, device, optimizer=None, switch_col_idx=None):
    training = optimizer is not None
    model.train(training)

    totals = defaultdict(float)
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        u = prepare_u(batch)

        load_edge_weight = None
        if switch_col_idx is not None:
            is_switch = batch.edge_attr[:, switch_col_idx] > 0.5
            load_edge_weight = torch.where(
                is_switch,
                torch.full_like(batch.edge_attr[:, 0], SWITCH_UPWEIGHT),
                torch.ones_like(batch.edge_attr[:, 0]),
            )

        with torch.set_grad_enabled(training):
            load_pred, scheduling_pred = model(
                batch.x, batch.edge_index, batch.edge_attr,
                batch=batch.batch, u=u,
            )
            load_target = (batch.y_load_residual if LOAD_TARGET == "residual"
                          else batch.y_load)
            losses = model.compute_losses(
                load_pred, scheduling_pred,
                load_target.float(),
                batch.y_scheduling.float(),
                batch.sched_mask.float(),
                scheduling_weight=LAMBDA_SCHED,
                load_edge_weight=load_edge_weight,
            )

        if training:
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        for k, v in losses.items():
            totals[k] += float(v)
        n_batches += 1

    return {k: v / max(1, n_batches) for k, v in totals.items()}


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    loads, scheds, y_l, y_r, y_s, masks, groups = \
        [], [], [], [], [], [], []

    for batch in loader:
        batch = batch.to(device)
        u = prepare_u(batch)
        load_pred, scheduling_pred = model(
            batch.x, batch.edge_index, batch.edge_attr,
            batch=batch.batch, u=u,
        )
        if LOAD_TARGET == "residual":
            load_pred = load_pred + batch.sp_load_log1p
        loads.append(load_pred.cpu())
        scheds.append(scheduling_pred.cpu())
        y_l.append(batch.y_load.cpu())
        y_r.append(batch.y_routing.cpu())
        y_s.append(batch.y_scheduling.cpu())
        masks.append(batch.sched_mask.cpu())

        # edge -> group mapping for per-topology breakdown
        src = batch.edge_index[0]
        graph_ids = batch.batch[src].cpu()
        # group_key is per-graph metadata (list under batching)
        gk = batch.group_key
        if isinstance(gk, str):
            gk = [gk]
        groups.extend(gk[int(g)] for g in graph_ids)

    return {
        "load_pred": torch.cat(loads),
        "sched_pred": torch.cat(scheds),
        "y_load": torch.cat(y_l).float(),
        "y_routing": torch.cat(y_r).float(),
        "y_scheduling": torch.cat(y_s).float(),
        "sched_mask": torch.cat(masks).float(),
        "groups": groups,
    }


@torch.no_grad()
def evaluate(preds, load_threshold=LOAD_USED_THRESHOLD):
    load_pred, y_load = preds["load_pred"], preds["y_load"]
    out = {}
    out.update({f"load_{k}": v
               for k, v in regression_metrics(load_pred, y_load).items()})

    # diagnostic only: used-edge detection derived from thresholding
    # load_pred, evaluated against the (near-degenerate on its own)
    # binary routing target -- NOT the metric that should drive model
    # selection, see module docstring fix #8.
    used_pred = (load_pred > load_threshold).float()
    out.update({f"used_edge_{k}": v for k, v in
               binary_metrics(used_pred, preds["y_routing"]).items()})

    used = preds["sched_mask"].bool()
    if used.any():
        err = preds["sched_pred"][used] - preds["y_scheduling"][used]
        out["sched_mae"] = float(err.abs().mean())
        out["sched_rmse"] = float((err ** 2).mean().sqrt())
    return out


@torch.no_grad()
def evaluate_per_group(preds, load_threshold=LOAD_USED_THRESHOLD):
    """Fix #5: per-topology test breakdown."""
    idx_by_group = defaultdict(list)
    for i, g in enumerate(preds["groups"]):
        idx_by_group[g].append(i)

    results = {}
    for g, idx in sorted(idx_by_group.items()):
        idx = torch.tensor(idx)
        sub = {
            "load_pred": preds["load_pred"][idx],
            "y_load": preds["y_load"][idx],
            "y_routing": preds["y_routing"][idx],
            "sched_pred": preds["sched_pred"][idx],
            "y_scheduling": preds["y_scheduling"][idx],
            "sched_mask": preds["sched_mask"][idx],
        }
        results[g] = evaluate(sub, load_threshold)
        results[g]["num_edges"] = int(idx.numel())
    return results


# ============================================================
# Single-seed run
# ============================================================

def train_one_seed(seed, train_data, val_data, test_data,
                   dims, exclude_keys, device):
    set_seed(seed)
    node_in_dim, edge_in_dim, global_in_dim = dims

    edge_feat_names = train_data[0].edge_feature_names
    switch_col_idx = (edge_feat_names.index(SWITCH_EDGE_FEAT_NAME)
                      if SWITCH_EDGE_FEAT_NAME in edge_feat_names else None)

    loader_kwargs = dict(
        pin_memory=(device.type == "cuda"),
        exclude_keys=exclude_keys,
    )
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                              shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE,
                            shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE,
                             shuffle=False, **loader_kwargs)

    model = StrongAgentGNN(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        global_in_dim=global_in_dim,
        hidden_dim=HIDDEN_DIM,
        heads=HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        use_graph_context=True,
        scheduling_activation=SCHEDULING_ACTIVATION,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8,
    )

    ckpt_path = MODEL_DIR / f"agent_best_seed{seed}.pt"
    history = defaultdict(list)
    best_val, best_epoch, patience = float("inf"), -1, 0

    for epoch in range(1, NUM_EPOCHS + 1):
        tr = run_epoch(model, train_loader, device, optimizer,
                       switch_col_idx=switch_col_idx)
        va = run_epoch(model, val_loader, device,
                       switch_col_idx=switch_col_idx)
        scheduler.step(va["total"])

        for k, v in tr.items():
            history[f"train_{k}"].append(v)
        for k, v in va.items():
            history[f"val_{k}"].append(v)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(f"[seed {seed}] Epoch {epoch:03d} | "
              f"train {tr['total']:.5f} "
              f"(L {tr['load']:.5f} S {tr['scheduling']:.5f}) | "
              f"val {va['total']:.5f} "
              f"(L {va['load']:.5f} S {va['scheduling']:.5f})")

        if va["total"] < best_val:
            best_val, best_epoch, patience = va["total"], epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": best_val,
                "seed": seed,
                "config": {
                    "node_in_dim": node_in_dim,
                    "edge_in_dim": edge_in_dim,
                    "global_in_dim": global_in_dim,
                    "hidden_dim": HIDDEN_DIM,
                    "heads": HEADS,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                    "scheduling_activation": SCHEDULING_ACTIVATION,
                    "lambda_sched": LAMBDA_SCHED,
                    "switch_upweight": SWITCH_UPWEIGHT,
                    "load_target_mode": LOAD_TARGET,
                },
            }, ckpt_path)
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"[seed {seed}] Early stop at {epoch} "
                      f"(best {best_epoch})")
                break

    # ---- restore best ----
    ckpt = torch.load(ckpt_path, map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # ---- test evaluation ----
    test_preds = collect_predictions(model, test_loader, device)
    results = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "test": evaluate(test_preds),
        "test_per_group": evaluate_per_group(test_preds),
    }

    torch.save(dict(history), MODEL_DIR / f"agent_history_seed{seed}.pt")
    return results


# ============================================================
# Aggregation
# ============================================================

def aggregate(per_seed_results):
    """mean ± std over seeds for every scalar test metric."""
    keys = per_seed_results[0]["test"].keys()
    agg = {"test": {}}
    for k in keys:
        vals = [r["test"][k] for r in per_seed_results
                if k in r["test"]]
        vals = [v for v in vals if v == v]  # drop NaN
        if not vals:
            continue
        t = torch.tensor(vals, dtype=torch.float)
        agg["test"][k] = {
            "mean": float(t.mean()),
            "std": float(t.std(unbiased=False)),
            "n_seeds": len(vals),
        }
    return agg


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_data = load_split(TRAIN_PATH, "Train")
    val_data = load_split(VAL_PATH, "Val")
    test_data = load_split(TEST_PATH, "Test")

    print(f"Train/Val/Test graphs: "
          f"{len(train_data)}/{len(val_data)}/{len(test_data)}")

    node_in_dim, edge_in_dim, global_in_dim = validate_schema(
        [train_data, val_data, test_data]
    )
    print(f"Dims — node: {node_in_dim}, edge: {edge_in_dim}, "
          f"u: {global_in_dim}")

    exclude_keys = detect_exclude_keys(train_data, val_data, test_data)
    # group_key must survive batching for per-topology evaluation
    if "group_key" in exclude_keys:
        exclude_keys.remove("group_key")

    per_seed = []
    for seed in SEEDS:
        print("\n" + "=" * 70)
        print(f"SEED {seed}")
        print("=" * 70)
        per_seed.append(
            train_one_seed(
                seed, train_data, val_data, test_data,
                (node_in_dim, edge_in_dim, global_in_dim),
                exclude_keys, device,
            )
        )

    summary = {
        "config": {
            "seeds": SEEDS,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "patience": PATIENCE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "hidden_dim": HIDDEN_DIM,
            "heads": HEADS,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "lambda_sched": LAMBDA_SCHED,
        },
        "per_seed": per_seed,
        "aggregate": aggregate(per_seed),
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("AGGREGATE (mean ± std over seeds)")
    print("=" * 70)
    for split, metrics in summary["aggregate"].items():
        print(f"\n{split}:")
        for k, v in metrics.items():
            print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")

    print("\nSaved:", RESULTS_PATH)


if __name__ == "__main__":
    main()
