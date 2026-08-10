"""
SimNet training script (thesis-ready).

Fixes vs. previous version
--------------------------
1. USES THE PIPELINE SPLITS: the old script re-split the FULL dataset
   randomly with sklearn, silently discarding the size-extrapolation /
   type-holdout splits and reintroducing leakage. We now load
   simnet_pyg_{train,val,test}.pt produced by build_simnet_pyg.py —
   the split is defined in exactly one place.
2. CORRECT INVERSE TRANSFORM: targets are log10(makespan); the old
   script inverted with np.exp (natural log), corrupting every
   real-scale metric. All inversions are 10**x, in one helper.
3. SINGLE TARGET-MODE CONTRACT: TARGET_MODE ('residual' | 'raw') is one
   config constant threaded into BOTH model.compute_loss and
   model.predict — the train/inference mismatch discussed earlier is
   structurally impossible. Residual mode learns y - log10(analytic
   lower bound), recommended for size extrapolation.
4. UNCERTAINTY: heteroscedastic NLL training (LOSS_TYPE='nll') and a
   sigma-calibration evaluation: corr(sigma, |error|) and empirical
   coverage of +-1sigma / +-1.96sigma intervals. This validates sigma
   as the trigger signal of the COCA calibration loop.
5. LOWER-BOUND BASELINE: every SimNet metric is reported next to the
   'predict the analytic lower bound' baseline. If SimNet does not
   beat it convincingly, the surrogate adds nothing — this comparison
   is the 'is SimNet necessary' answer the thesis must contain.
6. HONEST METRICS: log10-space MAE/RMSE + real-scale MAPE and median
   APE (real-scale MAE alone is dominated by the largest makespans),
   with per-node-count / per-topology-type / per-policy-type
   breakdowns on test.
7. Multi-seed with mean +- std aggregation, early stopping, LR
   scheduler, deterministic seeding, full-config checkpoints,
   headless-safe plotting. Same conventions as train_agent.py.
"""

import sys
import json
import math
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch_geometric.loader import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.simnet_gnn import SimNetGNN  # noqa: E402


# ============================================================
# Paths / config
# ============================================================

DATA_DIR = ROOT / "data" / "processed"
TRAIN_PATH = DATA_DIR / "simnet_pyg_train.pt"
VAL_PATH = DATA_DIR / "simnet_pyg_val.pt"
TEST_PATH = DATA_DIR / "simnet_pyg_test.pt"
STATS_PATH = DATA_DIR / "simnet_pyg_stats.pt"

MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "figures" / "simnet"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = MODEL_DIR / "simnet_results.json"

SEEDS = [42, 1337, 2024]
CASE_SEED = SEEDS[0]                # plots come from this seed

BATCH_SIZE = 32
NUM_EPOCHS = 250
PATIENCE = 30
LR = 5e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 2.0

HIDDEN_DIM = 128
HEADS = 4
NUM_LAYERS = 3
DROPOUT = 0.10

# ---- the single-source-of-truth contract (fix #3) ----
TARGET_MODE = "residual"            # 'residual' | 'raw'
LOSS_TYPE = "nll"                   # 'nll' | 'mse'

REQUIRED_KEYS = {"x", "edge_index", "edge_attr", "u", "y", "lb_log10"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def inv(y_log10):
    """The ONLY inverse transform in this file (fix #2)."""
    return np.power(10.0, np.asarray(y_log10, dtype=float))


# ============================================================
# Data
# ============================================================

def load_split(path, name):
    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found: {path} — run build_simnet_pyg.py."
        )
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not data:
        raise ValueError(f"{name} is empty.")
    return data


def attach_lb(dataset):
    """Store log10(LB) as a tensor attribute so PyG collates it into
    a [num_graphs] tensor per batch."""
    for g in dataset:
        lb = float(g.analytic_lower_bound)
        g.lb_log10 = torch.tensor(
            [math.log10(max(lb, 1e-12))], dtype=torch.float
        )
    return dataset


def validate(datasets):
    dims = {"node": set(), "edge": set(), "u": set()}
    for split in datasets:
        for g in split:
            dims["node"].add(g.x.size(1))
            dims["edge"].add(g.edge_attr.size(1))
            dims["u"].add(g.u.size(-1))
            assert g.y.numel() == 1
            assert g.lb_log10.numel() == 1
            assert torch.isfinite(g.y).all()
    for k, v in dims.items():
        if len(v) != 1:
            raise ValueError(
                f"Inconsistent {k} dims {v} — fix the pipeline."
            )
    return (dims["node"].pop(), dims["edge"].pop(), dims["u"].pop())


def detect_exclude_keys(*datasets):
    all_keys = set()
    for split in datasets:
        for g in split[:50]:
            all_keys.update(g.keys())
    return sorted(all_keys - REQUIRED_KEYS)


def lb_arg(batch):
    """lb passed to the model only in residual mode (fix #3)."""
    return batch.lb_log10.view(-1) if TARGET_MODE == "residual" else None


# ============================================================
# Epochs
# ============================================================

def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, n = 0.0, 0

    for batch in loader:
        batch = batch.to(device)
        with torch.set_grad_enabled(training):
            mean, log_var = model(batch.x, batch.edge_index,
                                  batch.edge_attr, batch.batch,
                                  batch.u)
            loss = model.compute_loss(
                mean, log_var, batch.y.view(-1),
                lb_log10=lb_arg(batch), loss_type=LOSS_TYPE,
            )
        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           GRAD_CLIP)
            optimizer.step()
        total += float(loss) * batch.num_graphs
        n += batch.num_graphs
    return total / max(1, n)


@torch.no_grad()
def collect_predictions(model, dataset, device):
    """Per-graph predictions with metadata (order-preserving)."""
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        exclude_keys=detect_exclude_keys(dataset))
    preds, sigmas, ys, lbs = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        pred, sigma = model.predict(
            batch.x, batch.edge_index, batch.edge_attr,
            batch.batch, batch.u, lb_log10=lb_arg(batch),
        )
        preds.append(pred.cpu())
        sigmas.append(sigma.cpu())
        ys.append(batch.y.view(-1).cpu())
        lbs.append(batch.lb_log10.view(-1).cpu())

    return {
        "pred_log": torch.cat(preds).numpy(),
        "sigma": torch.cat(sigmas).numpy(),
        "y_log": torch.cat(ys).numpy(),
        "lb_log": torch.cat(lbs).numpy(),
        "num_nodes": np.array([g.num_nodes_meta for g in dataset]),
        "topology_type": np.array([g.topology_type for g in dataset]),
        "policy_type": np.array([g.policy_type for g in dataset]),
    }


# ============================================================
# Metrics
# ============================================================

def regression_metrics(pred_log, y_log):
    err_log = pred_log - y_log
    pred, y = inv(pred_log), inv(y_log)
    ape = np.abs(pred - y) / np.maximum(y, 1e-12)
    return {
        "log10_mae": float(np.abs(err_log).mean()),
        "log10_rmse": float(np.sqrt((err_log ** 2).mean())),
        "mape": float(ape.mean()),
        "median_ape": float(np.median(ape)),
        "n": int(len(y)),
    }


def sigma_calibration(pred_log, sigma, y_log):
    """Is sigma a valid reliability signal? (fix #4)"""
    err = np.abs(pred_log - y_log)
    if sigma.std() < 1e-12 or err.std() < 1e-12:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(sigma, err)[0, 1])
    cov68 = float((err <= sigma).mean())
    cov95 = float((err <= 1.96 * sigma).mean())
    return {
        "corr_sigma_abs_error": corr,
        "coverage_1sigma": cov68,      # target ~0.68
        "coverage_1.96sigma": cov95,   # target ~0.95
        "mean_sigma": float(sigma.mean()),
    }


def breakdown(preds, key):
    out = {}
    vals = preds[key]
    for v in sorted(set(vals.tolist())):
        m = vals == v
        out[str(v)] = regression_metrics(
            preds["pred_log"][m], preds["y_log"][m]
        )
    return out


def evaluate(preds):
    res = {
        "simnet": regression_metrics(preds["pred_log"],
                                     preds["y_log"]),
        # fix #5: 'predict the analytic lower bound' baseline
        "lb_baseline": regression_metrics(preds["lb_log"],
                                          preds["y_log"]),
        "sigma_calibration": sigma_calibration(
            preds["pred_log"], preds["sigma"], preds["y_log"]
        ),
        "by_num_nodes": breakdown(preds, "num_nodes"),
        "by_topology_type": breakdown(preds, "topology_type"),
        "by_policy_type": breakdown(preds, "policy_type"),
    }
    res["mape_improvement_over_lb"] = (
        res["lb_baseline"]["mape"] / max(1e-12, res["simnet"]["mape"])
    )
    return res


# ============================================================
# Plots (case seed only)
# ============================================================

def plot_curve(history, tag):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(history["train"], lw=2, label="Train loss")
    ax.plot(history["val"], lw=2, label="Val loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel(f"{LOSS_TYPE} loss")
    ax.set_title(f"SimNet training ({TARGET_MODE} target)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"training_curve_{tag}.png", dpi=300)
    plt.close(fig)


def plot_pred_vs_true(preds, tag):
    y, p = inv(preds["y_log"]), inv(preds["pred_log"])
    lb = inv(preds["lb_log"])
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(y, lb, s=18, alpha=0.4, c="#BBBBBB",
               label="LB baseline")
    ax.scatter(y, p, s=18, alpha=0.6, c="#2C7FB8", label="SimNet")
    lo = min(y.min(), p.min()) * 0.8
    hi = max(y.max(), p.max()) * 1.2
    ax.plot([lo, hi], [lo, hi], "--", c="black", lw=1.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("True makespan (s, log)")
    ax.set_ylabel("Predicted makespan (s, log)")
    ax.set_title("SimNet vs LB baseline (test)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"pred_vs_true_{tag}.png", dpi=300)
    plt.close(fig)


def plot_sigma_calibration(preds, tag):
    err = np.abs(preds["pred_log"] - preds["y_log"])
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    ax.scatter(preds["sigma"], err, s=16, alpha=0.5, c="#E45756")
    m = max(preds["sigma"].max(), err.max())
    ax.plot([0, m], [0, m], "--", c="black", lw=1.2,
            label="|error| = sigma")
    ax.set_xlabel("Predicted sigma (log10 space)")
    ax.set_ylabel("|prediction error| (log10 space)")
    ax.set_title("Uncertainty calibration: sigma vs realized error")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"sigma_calibration_{tag}.png", dpi=300)
    plt.close(fig)


def plot_size_breakdown(res, tag):
    sizes = sorted(res["by_num_nodes"].keys(), key=int)
    mapes = [res["by_num_nodes"][s]["mape"] for s in sizes]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.bar(sizes, mapes, color="#9ECAE1", edgecolor="black")
    ax.axhline(res["lb_baseline"]["mape"], ls="--", c="black",
               label="LB baseline (overall)")
    ax.set_xlabel("Number of nodes (test)")
    ax.set_ylabel("MAPE")
    ax.set_title("Size-extrapolation performance")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"size_breakdown_{tag}.png", dpi=300)
    plt.close(fig)


# ============================================================
# One seed
# ============================================================

def train_one_seed(seed, splits, dims, exclude_keys, device):
    set_seed(seed)
    train_data, val_data, test_data = splits
    node_in, edge_in, global_in = dims

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                              shuffle=True,
                              exclude_keys=exclude_keys)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE,
                            shuffle=False, exclude_keys=exclude_keys)

    model = SimNetGNN(
        node_in_dim=node_in, edge_in_dim=edge_in,
        global_in_dim=global_in,
        hidden_dim=HIDDEN_DIM, heads=HEADS,
        num_layers=NUM_LAYERS, dropout=DROPOUT,
        predict_uncertainty=(LOSS_TYPE == "nll"),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )

    ckpt_path = MODEL_DIR / f"simnet_best_seed{seed}.pt"
    history = {"train": [], "val": []}
    best_val, best_epoch, patience = float("inf"), -1, 0

    for epoch in range(1, NUM_EPOCHS + 1):
        tr = run_epoch(model, train_loader, device, optimizer)
        va = run_epoch(model, val_loader, device)
        scheduler.step(va)
        history["train"].append(tr)
        history["val"].append(va)

        if epoch % 10 == 0 or epoch == 1:
            print(f"[seed {seed}] Epoch {epoch:03d} | "
                  f"train {tr:.5f} | val {va:.5f}")

        if va < best_val:
            best_val, best_epoch, patience = va, epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "val_loss": best_val, "seed": seed,
                "config": {
                    "node_in_dim": node_in, "edge_in_dim": edge_in,
                    "global_in_dim": global_in,
                    "hidden_dim": HIDDEN_DIM, "heads": HEADS,
                    "num_layers": NUM_LAYERS, "dropout": DROPOUT,
                    "target_mode": TARGET_MODE,
                    "loss_type": LOSS_TYPE,
                },
            }, ckpt_path)
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"[seed {seed}] early stop at {epoch} "
                      f"(best {best_epoch})")
                break

    ckpt = torch.load(ckpt_path, map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    preds = collect_predictions(model, test_data, device)
    res = evaluate(preds)
    res["seed"] = seed
    res["best_epoch"] = best_epoch
    res["best_val_loss"] = best_val

    if seed == CASE_SEED:
        tag = f"seed{seed}"
        plot_curve(history, tag)
        plot_pred_vs_true(preds, tag)
        if LOSS_TYPE == "nll":
            plot_sigma_calibration(preds, tag)
        plot_size_breakdown(res, tag)

    return res


# ============================================================
# Aggregate + main
# ============================================================

def aggregate(per_seed):
    agg = {}
    for block in ("simnet", "lb_baseline"):
        agg[block] = {}
        for k in per_seed[0][block]:
            if k == "n":
                continue
            vals = np.array([r[block][k] for r in per_seed])
            agg[block][k] = {"mean": float(vals.mean()),
                             "std": float(vals.std())}
    vals = np.array([r["mape_improvement_over_lb"] for r in per_seed])
    agg["mape_improvement_over_lb"] = {
        "mean": float(vals.mean()), "std": float(vals.std())
    }
    return agg


def main():
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    print("Device:", device)
    print(f"TARGET_MODE={TARGET_MODE}  LOSS_TYPE={LOSS_TYPE}")

    train_data = attach_lb(load_split(TRAIN_PATH, "Train"))
    val_data = attach_lb(load_split(VAL_PATH, "Val"))
    test_data = attach_lb(load_split(TEST_PATH, "Test"))
    print(f"Train/Val/Test: {len(train_data)}/{len(val_data)}"
          f"/{len(test_data)}")

    dims = validate([train_data, val_data, test_data])
    exclude_keys = detect_exclude_keys(train_data, val_data,
                                       test_data)

    per_seed = []
    for seed in SEEDS:
        print("\n" + "=" * 70)
        print(f"SEED {seed}")
        print("=" * 70)
        per_seed.append(train_one_seed(
            seed, (train_data, val_data, test_data),
            dims, exclude_keys, device,
        ))

    summary = {
        "config": {
            "seeds": SEEDS, "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS, "patience": PATIENCE,
            "lr": LR, "weight_decay": WEIGHT_DECAY,
            "hidden_dim": HIDDEN_DIM, "heads": HEADS,
            "num_layers": NUM_LAYERS, "dropout": DROPOUT,
            "target_mode": TARGET_MODE, "loss_type": LOSS_TYPE,
        },
        "per_seed": per_seed,
        "aggregate": aggregate(per_seed),
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n" + "=" * 70)
    print("AGGREGATE (mean ± std over seeds)")
    for block in ("simnet", "lb_baseline"):
        print(f"\n{block}:")
        for k, v in summary["aggregate"][block].items():
            print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")
    imp = summary["aggregate"]["mape_improvement_over_lb"]
    print(f"\nMAPE improvement over LB baseline: "
          f"{imp['mean']:.2f}x ± {imp['std']:.2f}")
    print("\nSaved:", RESULTS_PATH)
    print("Figures in:", FIG_DIR)


if __name__ == "__main__":
    main()
