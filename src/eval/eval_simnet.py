"""
SimNet evaluation suite (thesis-grade).

Replaces the old eval + analysis pair. What was wrong before:
  - evaluated on the FULL dataset (training samples included) — every
    reported number was optimistically biased;
  - inverted log targets with np.exp (targets are log10);
  - guessed checkpoint formats and forward signatures defensively
    instead of relying on one strict contract;
  - no residual-target handling, no uncertainty analysis, no baseline,
    and nothing about the surrogate's actual job in the COCA loop.

What this suite answers:

Q1  SIZE EXTRAPOLATION (the thesis claim): error vs num_nodes across
    ALL splits, clearly labeled in-distribution (train sizes) vs
    extrapolation (val/test sizes). Test numbers come from test only;
    the cross-split plot exists to SHOW the generalization gap, not to
    hide it.

Q2  RANKING FIDELITY (the surrogate's real job): in COCA step 4 SimNet
    scores candidate schedules — it must ORDER them correctly far more
    than it must predict absolute makespans. We report global Spearman
    correlation and, more importantly, within-group pairwise
    concordance: among schedules for the SAME (topology type, size,
    chunk) setting, how often does SimNet pick the faster one?
    Compared against the LB baseline's concordance.

Q3  UNCERTAINTY AS AN OOD / CALIBRATION-TRIGGER SIGNAL: does predicted
    sigma (a) track realized error (corr + coverage), and (b) RISE on
    the extrapolation splits relative to validation? If yes, sigma is
    a valid trigger for 'query the real simulator' in the calibration
    loop.

Q4  WHERE IS IT WEAK: breakdowns by chunk size, topology type, and
    policy type — with special attention to 'detour' and
    'allgather_tree' (the schedule-like inputs SimNet will actually
    score), all against the analytic-lower-bound baseline.

Q5  END-TO-END CASE STUDY: one experimental setting with several
    candidate schedules, true vs predicted makespans side by side with
    sigma error bars — the 'what does SimNet actually do' figure.

Multi-seed: metrics aggregated over seeds; plots from CASE_SEED.
Contract: checkpoints from train_simnet.py (config incl. target_mode).
"""

import sys
import json
import math
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.models.simnet_gnn import SimNetGNN  # noqa: E402


# ============================================================
# Paths / config
# ============================================================

DATA_DIR = BASE_DIR / "data" / "processed"
SPLIT_PATHS = {
    "train": DATA_DIR / "simnet_pyg_train.pt",
    "val": DATA_DIR / "simnet_pyg_val.pt",
    "test": DATA_DIR / "simnet_pyg_test.pt",
}

MODEL_DIR = BASE_DIR / "models"
EVAL_DIR = BASE_DIR / "src" / "eval"
PLOT_DIR = EVAL_DIR / "plots" / "simnet"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 1337, 2024]
CASE_SEED = SEEDS[0]
BATCH_SIZE = 64

SUMMARY_PATH = EVAL_DIR / "simnet_eval_summary.json"
CSV_PATH = EVAL_DIR / "simnet_eval_per_graph.csv"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 14, "figure.dpi": 120, "savefig.dpi": 300,
})

SPLIT_COLORS = {"train": "#54A24B", "val": "#F58518",
                "test": "#E45756"}


def clean_axes(ax):
    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(PLOT_DIR / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", PLOT_DIR / name)


def inv(y_log10):
    """log10 -> seconds. The ONLY inverse transform here."""
    return np.power(10.0, np.asarray(y_log10, dtype=float))


# ============================================================
# Loading
# ============================================================

def load_model(seed, device):
    path = MODEL_DIR / f"simnet_best_seed{seed}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path} — run train_simnet.py."
        )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = SimNetGNN(
        node_in_dim=cfg["node_in_dim"],
        edge_in_dim=cfg["edge_in_dim"],
        global_in_dim=cfg["global_in_dim"],
        hidden_dim=cfg["hidden_dim"],
        heads=cfg["heads"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        predict_uncertainty=(cfg["loss_type"] == "nll"),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg["target_mode"]      # contract, not a guess


def load_splits():
    splits = {}
    for name, path in SPLIT_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{name} split missing: {path} — "
                f"run build_simnet_pyg.py."
            )
        data = torch.load(path, map_location="cpu",
                          weights_only=False)
        for g in data:
            lb = float(g.analytic_lower_bound)
            g.lb_log10 = torch.tensor(
                [math.log10(max(lb, 1e-12))], dtype=torch.float
            )
        splits[name] = data
    return splits


REQUIRED_KEYS = {"x", "edge_index", "edge_attr", "u", "y", "lb_log10"}


def exclude_keys(dataset):
    keys = set()
    for g in dataset[:50]:
        keys.update(g.keys())
    return sorted(keys - REQUIRED_KEYS)


@torch.no_grad()
def collect(model, dataset, target_mode, device):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        exclude_keys=exclude_keys(dataset))
    preds, sigmas, ys, lbs = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        lb = (batch.lb_log10.view(-1)
              if target_mode == "residual" else None)
        p, s = model.predict(batch.x, batch.edge_index,
                             batch.edge_attr, batch.batch, batch.u,
                             lb_log10=lb)
        preds.append(p.cpu()); sigmas.append(s.cpu())
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
        "chunk_mb": np.array([g.chunk_mb for g in dataset]),
        "sample_id": np.array([g.sample_id for g in dataset]),
    }


# ============================================================
# Metrics
# ============================================================

def reg_metrics(pred_log, y_log):
    err = pred_log - y_log
    ape = np.abs(inv(pred_log) - inv(y_log)) / np.maximum(
        inv(y_log), 1e-12)
    return {"log10_mae": float(np.abs(err).mean()),
            "log10_rmse": float(np.sqrt((err ** 2).mean())),
            "mape": float(ape.mean()),
            "median_ape": float(np.median(ape)),
            "n": int(len(y_log))}


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def group_concordance(preds, score_key):
    """
    Q2: within groups of the SAME (topology_type, num_nodes, chunk_mb)
    setting, over all pairs with distinct true makespans: how often
    does `score_key` order the pair the same way as the truth?
    """
    groups = defaultdict(list)
    for i in range(len(preds["y_log"])):
        key = (preds["topology_type"][i],
               int(preds["num_nodes"][i]),
               float(preds["chunk_mb"][i]))
        groups[key].append(i)

    concordant, total, usable_groups = 0, 0, 0
    for key, idx in groups.items():
        if len(idx) < 2:
            continue
        usable_groups += 1
        for i, j in combinations(idx, 2):
            dt = preds["y_log"][i] - preds["y_log"][j]
            if abs(dt) < 1e-9:
                continue
            dp = preds[score_key][i] - preds[score_key][j]
            total += 1
            if np.sign(dp) == np.sign(dt):
                concordant += 1

    return {
        "pairwise_concordance": (concordant / total
                                 if total else float("nan")),
        "n_pairs": total,
        "n_groups_with_2plus": usable_groups,
    }


def sigma_analysis(preds):
    err = np.abs(preds["pred_log"] - preds["y_log"])
    s = preds["sigma"]
    corr = (float(np.corrcoef(s, err)[0, 1])
            if s.std() > 1e-12 and err.std() > 1e-12
            else float("nan"))
    return {"corr_sigma_abs_error": corr,
            "coverage_1sigma": float((err <= s).mean()),
            "coverage_1.96sigma": float((err <= 1.96 * s).mean()),
            "mean_sigma": float(s.mean())}


def breakdown(preds, key, with_lb=True):
    out = {}
    for v in sorted(set(preds[key].tolist())):
        m = preds[key] == v
        out[str(v)] = reg_metrics(preds["pred_log"][m],
                                  preds["y_log"][m])
        if with_lb:
            out[str(v)]["lb_mape"] = reg_metrics(
                preds["lb_log"][m], preds["y_log"][m])["mape"]
    return out


# ============================================================
# Plots (case seed)
# ============================================================

def plot_size_extrapolation(all_preds):
    """Q1 — the thesis plot: MAPE vs n, colored by split."""
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for split, preds in all_preds.items():
        sizes = sorted(set(preds["num_nodes"].tolist()))
        xs, ys_m, ys_lb = [], [], []
        for n in sizes:
            m = preds["num_nodes"] == n
            xs.append(n)
            ys_m.append(reg_metrics(preds["pred_log"][m],
                                    preds["y_log"][m])["mape"])
            ys_lb.append(reg_metrics(preds["lb_log"][m],
                                     preds["y_log"][m])["mape"])
        ax.plot(xs, ys_m, "o-", lw=2.2, color=SPLIT_COLORS[split],
                label=f"SimNet ({split})")
        ax.plot(xs, ys_lb, "s--", lw=1.4, color=SPLIT_COLORS[split],
                alpha=0.5, label=f"LB baseline ({split})")
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("MAPE")
    ax.set_title("Q1 — Size extrapolation: error vs topology size\n"
                 "(train sizes = in-distribution; "
                 "val/test sizes = extrapolation)")
    ax.legend(fontsize=8)
    clean_axes(ax)
    savefig(fig, "q1_size_extrapolation.png")


def plot_pred_vs_true(preds, tag="test"):
    y, p, lb = inv(preds["y_log"]), inv(preds["pred_log"]), \
        inv(preds["lb_log"])
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(y, lb, s=16, alpha=0.35, c="#BBBBBB",
               label="LB baseline")
    ax.scatter(y, p, s=16, alpha=0.6, c="#2C7FB8", label="SimNet")
    lo = min(y.min(), p.min()) * 0.7
    hi = max(y.max(), p.max()) * 1.4
    ax.plot([lo, hi], [lo, hi], "--", c="black", lw=1.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("True makespan (s)")
    ax.set_ylabel("Predicted makespan (s)")
    ax.set_title(f"Predicted vs true ({tag} split)")
    ax.legend(); clean_axes(ax)
    savefig(fig, f"pred_vs_true_{tag}.png")


def plot_chunk_effect(preds):
    """Q4 — signed residual (log space) vs chunk size, with trend."""
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    res = preds["pred_log"] - preds["y_log"]
    ax.scatter(preds["chunk_mb"], res, s=22, alpha=0.4, c="#2C7FB8")
    trend_x, trend_y = [], []
    for c in sorted(set(preds["chunk_mb"].tolist())):
        m = preds["chunk_mb"] == c
        trend_x.append(c); trend_y.append(float(res[m].mean()))
    ax.plot(trend_x, trend_y, "o-", c="black", lw=2.2,
            label="Mean residual")
    ax.axhline(0, ls="--", c="black", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("Chunk size (MB, log)")
    ax.set_ylabel("Residual in log10 space (pred − true)")
    ax.set_title("Q4 — Systematic bias vs chunk size (test)")
    ax.legend(); clean_axes(ax)
    savefig(fig, "q4_residual_vs_chunk_size.png")


def plot_policy_breakdown(preds):
    """Q4 — the surrogate on the inputs it will actually score."""
    ptypes = sorted(set(preds["policy_type"].tolist()))
    mapes, lb_mapes = [], []
    for p in ptypes:
        m = preds["policy_type"] == p
        mapes.append(reg_metrics(preds["pred_log"][m],
                                 preds["y_log"][m])["mape"])
        lb_mapes.append(reg_metrics(preds["lb_log"][m],
                                    preds["y_log"][m])["mape"])
    x = np.arange(len(ptypes)); w = 0.38
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.bar(x - w / 2, mapes, w, label="SimNet", color="#2C7FB8",
           edgecolor="black")
    ax.bar(x + w / 2, lb_mapes, w, label="LB baseline",
           color="#BBBBBB", edgecolor="black")
    # highlight the schedule-like policies
    for i, p in enumerate(ptypes):
        if p in ("detour", "allgather_tree"):
            ax.axvspan(i - 0.5, i + 0.5, color="#FDD0A2", alpha=0.25)
    ax.set_xticks(x); ax.set_xticklabels(ptypes, rotation=20)
    ax.set_ylabel("MAPE")
    ax.set_title("Q4 — Error by policy type (test). Shaded = "
                 "schedule-like inputs SimNet will actually score.")
    ax.legend(); clean_axes(ax)
    savefig(fig, "q4_policy_type_breakdown.png")


def plot_sigma_ood(all_preds):
    """Q3b — does sigma rise on extrapolation splits?"""
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    names, data = [], []
    for split in ("train", "val", "test"):
        names.append(split)
        data.append(all_preds[split]["sigma"])
    bp = ax.boxplot(data, labels=names, patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color="black", lw=1.4))
    for patch, name in zip(bp["boxes"], names):
        patch.set_facecolor(SPLIT_COLORS[name]); patch.set_alpha(0.5)
    ax.set_ylabel("Predicted sigma (log10 space)")
    ax.set_title("Q3b — Uncertainty as an OOD signal:\n"
                 "sigma should rise from train sizes to "
                 "extrapolation sizes")
    clean_axes(ax)
    savefig(fig, "q3b_sigma_ood_shift.png")


def plot_sigma_reliability(preds):
    err = np.abs(preds["pred_log"] - preds["y_log"])
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    ax.scatter(preds["sigma"], err, s=16, alpha=0.5, c="#E45756")
    m = max(preds["sigma"].max(), err.max())
    ax.plot([0, m], [0, m], "--", c="black", lw=1.2,
            label="|error| = sigma")
    sa = sigma_analysis(preds)
    ax.text(0.03, 0.95,
            f"corr = {sa['corr_sigma_abs_error']:.3f}\n"
            f"cov@1σ = {sa['coverage_1sigma']:.2f} (→0.68)\n"
            f"cov@1.96σ = {sa['coverage_1.96sigma']:.2f} (→0.95)",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="black"))
    ax.set_xlabel("Predicted sigma")
    ax.set_ylabel("|error| (log10 space)")
    ax.set_title("Q3a — Sigma vs realized error (test)")
    ax.legend(); clean_axes(ax)
    savefig(fig, "q3a_sigma_reliability.png")


def plot_case_study(preds):
    """Q5 — one setting, several candidate schedules, ranked."""
    groups = defaultdict(list)
    for i in range(len(preds["y_log"])):
        key = (preds["topology_type"][i],
               int(preds["num_nodes"][i]),
               float(preds["chunk_mb"][i]))
        groups[key].append(i)

    # the group with the most candidates makes the best figure
    key, idx = max(groups.items(), key=lambda kv: len(kv[1]))
    if len(idx) < 3:
        print("Case study skipped: no group with >=3 candidates.")
        return

    idx = sorted(idx, key=lambda i: preds["y_log"][i])
    labels = [str(preds["policy_type"][i]) for i in idx]
    y = inv(preds["y_log"][idx])
    p = inv(preds["pred_log"][idx])
    s_hi = inv(preds["pred_log"][idx] + preds["sigma"][idx]) - p
    s_lo = p - inv(preds["pred_log"][idx] - preds["sigma"][idx])

    x = np.arange(len(idx)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(8.0, 1.3 * len(idx)), 5.4))
    ax.bar(x - w / 2, y, w, label="True (simulator)",
           color="#BBBBBB", edgecolor="black")
    ax.bar(x + w / 2, p, w, yerr=[s_lo, s_hi], capsize=4,
           label="SimNet (±1σ)", color="#2C7FB8", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Makespan (s)")
    ax.set_yscale("log")
    conc = group_concordance(
        {k: (v[idx] if isinstance(v, np.ndarray) else v)
         for k, v in preds.items()}, "pred_log"
    )["pairwise_concordance"]
    ax.set_title(
        f"Q5 — Case study: {key[0]}, n={key[1]}, {key[2]:.0f} MB\n"
        f"{len(idx)} candidate schedules, sorted by true makespan "
        f"(within-group concordance here: {conc:.0%})"
    )
    ax.legend(); clean_axes(ax)
    savefig(fig, "q5_case_study_schedule_ranking.png")


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    print("Device:", device)

    splits = load_splits()
    for name, d in splits.items():
        print(f"{name}: {len(d)} graphs")

    all_seed_summaries = {}

    for seed in SEEDS:
        try:
            model, target_mode = load_model(seed, device)
        except FileNotFoundError as e:
            print("SKIP seed", seed, "-", e)
            continue

        print(f"\n===== seed {seed} "
              f"(target_mode={target_mode}) =====")

        all_preds = {name: collect(model, d, target_mode, device)
                     for name, d in splits.items()}
        test = all_preds["test"]

        summary = {
            "target_mode": target_mode,
            # headline numbers: TEST ONLY — never the full dataset
            "test_simnet": reg_metrics(test["pred_log"],
                                       test["y_log"]),
            "test_lb_baseline": reg_metrics(test["lb_log"],
                                            test["y_log"]),
            "test_spearman": spearman(test["pred_log"],
                                      test["y_log"]),
            "test_ranking_simnet": group_concordance(test,
                                                     "pred_log"),
            "test_ranking_lb": group_concordance(test, "lb_log"),
            "test_sigma": sigma_analysis(test),
            "sigma_by_split": {
                s: float(all_preds[s]["sigma"].mean())
                for s in all_preds
            },
            "test_by_chunk_mb": breakdown(test, "chunk_mb"),
            "test_by_topology_type": breakdown(test,
                                               "topology_type"),
            "test_by_policy_type": breakdown(test, "policy_type"),
        }
        summary["mape_improvement_over_lb"] = (
            summary["test_lb_baseline"]["mape"]
            / max(1e-12, summary["test_simnet"]["mape"])
        )
        all_seed_summaries[seed] = summary

        print(f"  test MAPE: {summary['test_simnet']['mape']:.4f} "
              f"(LB: {summary['test_lb_baseline']['mape']:.4f}, "
              f"{summary['mape_improvement_over_lb']:.2f}x better)")
        print(f"  test Spearman: {summary['test_spearman']:.4f}")
        rk = summary["test_ranking_simnet"]
        rk_lb = summary["test_ranking_lb"]
        print(f"  pairwise concordance: "
              f"SimNet {rk['pairwise_concordance']:.3f} vs "
              f"LB {rk_lb['pairwise_concordance']:.3f} "
              f"({rk['n_pairs']} pairs, "
              f"{rk['n_groups_with_2plus']} groups)")
        print(f"  sigma by split: {summary['sigma_by_split']}")

        if seed == CASE_SEED:
            plot_size_extrapolation(all_preds)
            plot_pred_vs_true(test, "test")
            plot_chunk_effect(test)
            plot_policy_breakdown(test)
            plot_sigma_reliability(test)
            plot_sigma_ood(all_preds)
            plot_case_study(test)

            # per-graph CSV (test only)
            pd.DataFrame({
                "sample_id": test["sample_id"],
                "topology_type": test["topology_type"],
                "policy_type": test["policy_type"],
                "num_nodes": test["num_nodes"],
                "chunk_mb": test["chunk_mb"],
                "true_makespan": inv(test["y_log"]),
                "pred_makespan": inv(test["pred_log"]),
                "lb_makespan": inv(test["lb_log"]),
                "sigma_log10": test["sigma"],
            }).to_csv(CSV_PATH, index=False)
            print("Saved:", CSV_PATH)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(all_seed_summaries, f, indent=2, default=float)
    print("\nSaved:", SUMMARY_PATH)
    print("Plots in:", PLOT_DIR)


if __name__ == "__main__":
    main()
