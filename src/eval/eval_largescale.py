import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[2]
EVAL_DIR = BASE_DIR / "src" / "eval"
PLOT_DIR = EVAL_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = EVAL_DIR / "simnet_graph_eval_rows.csv"


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 120,
    "savefig.dpi": 300,
})


def clean_axes(ax):
    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(PLOT_DIR / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def safe_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None

    return float(np.corrcoef(x, y)[0, 1])


def rmse(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x ** 2)))


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing CSV: {CSV_PATH}. Run evaluate_simnet.py first.")

    df = pd.read_csv(CSV_PATH)

    required = ["true_time", "pred_time", "topology_type", "message_mb"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}. Available: {list(df.columns)}")

    d = df.copy()
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=required)

    d["residual"] = d["pred_time"] - d["true_time"]
    d["abs_error"] = np.abs(d["residual"])
    d["relative_error"] = d["abs_error"] / np.maximum(np.abs(d["true_time"]), 1e-12)

    # ========================================================
    # Completion-time regimes
    # ========================================================

    q1, q2 = d["true_time"].quantile([1 / 3, 2 / 3])

    def assign_regime(x):
        if x <= q1:
            return "Low completion time"
        if x <= q2:
            return "Medium completion time"
        return "High completion time"

    d["completion_regime"] = d["true_time"].apply(assign_regime)

    regime_order = [
        "Low completion time",
        "Medium completion time",
        "High completion time",
    ]

    summary_regime = (
        d.groupby("completion_regime")
        .agg(
            n=("true_time", "count"),
            true_time_mean=("true_time", "mean"),
            abs_error_mean=("abs_error", "mean"),
            residual_mean=("residual", "mean"),
            residual_std=("residual", "std"),
            rmse=("residual", rmse),
            relative_error_mean=("relative_error", "mean"),
        )
        .reindex(regime_order)
        .reset_index()
    )

    summary_topology_regime = (
        d.groupby(["topology_type", "completion_regime"])
        .agg(
            n=("true_time", "count"),
            abs_error_mean=("abs_error", "mean"),
            residual_mean=("residual", "mean"),
            rmse=("residual", rmse),
            relative_error_mean=("relative_error", "mean"),
        )
        .reset_index()
    )

    corr_abs = safe_pearson(d["true_time"], d["abs_error"])
    corr_rel = safe_pearson(d["true_time"], d["relative_error"])
    corr_res = safe_pearson(d["true_time"], d["residual"])

    summary = {
        "num_samples": int(len(d)),
        "completion_time_quantiles": {
            "q33": float(q1),
            "q66": float(q2),
        },
        "correlations": {
            "pearson_true_time_vs_abs_error": corr_abs,
            "pearson_true_time_vs_relative_error": corr_rel,
            "pearson_true_time_vs_residual": corr_res,
        },
    }

    out_csv_1 = EVAL_DIR / "simnet_large_scale_summary_by_regime.csv"
    out_csv_2 = EVAL_DIR / "simnet_large_scale_summary_by_topology_regime.csv"
    out_json = EVAL_DIR / "simnet_large_scale_analysis_summary.json"

    summary_regime.to_csv(out_csv_1, index=False)
    summary_topology_regime.to_csv(out_csv_2, index=False)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSummary by completion-time regime:")
    print(summary_regime)

    print("\nCorrelation summary:")
    print(json.dumps(summary, indent=2))

    # ========================================================
    # Plot 1 — Absolute Error vs True Completion Time
    # ========================================================

    fig, ax = plt.subplots(figsize=(8.7, 5.3))

    topologies = sorted(d["topology_type"].unique())
    palette = {
        "line": "#4C78A8",
        "ring": "#F58518",
        "star": "#54A24B",
        "bottleneck": "#E45756",
        "random": "#72B7B2",
    }

    for topo in topologies:
        sub = d[d["topology_type"] == topo]
        ax.scatter(
            sub["true_time"],
            sub["abs_error"],
            s=42,
            alpha=0.72,
            facecolors="white",
            edgecolors=palette.get(topo, "#555555"),
            linewidths=1.1,
            label=topo,
        )

    # Mean trend by true-time bins
    d["true_time_bin"] = pd.qcut(d["true_time"], q=8, duplicates="drop")
    trend = (
        d.groupby("true_time_bin", observed=True)
        .agg(
            true_time_mean=("true_time", "mean"),
            abs_error_mean=("abs_error", "mean"),
        )
        .reset_index()
        .sort_values("true_time_mean")
    )

    ax.plot(
        trend["true_time_mean"],
        trend["abs_error_mean"],
        color="black",
        linewidth=2.8,
        marker="o",
        markersize=5,
        label="Binned mean trend",
        zorder=8,
    )

    corr_text = "Pearson r unavailable" if corr_abs is None else f"Pearson r = {corr_abs:.3f}"

    ax.text(
        0.97,
        0.08,
        corr_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.92),
    )

    ax.set_title("SimNet Absolute Error vs True Completion Time")
    ax.set_xlabel("True Completion Time")
    ax.set_ylabel("Absolute Completion Time Error")
    clean_axes(ax)
    ax.legend(fontsize=8, frameon=True, loc="upper right")
    savefig(fig, "simnet_large_01_abs_error_vs_true_time.png")

    # ========================================================
    # Plot 2 — Residual vs True Completion Time
    # ========================================================

    fig, ax = plt.subplots(figsize=(8.7, 5.3))

    for topo in topologies:
        sub = d[d["topology_type"] == topo]
        ax.scatter(
            sub["true_time"],
            sub["residual"],
            s=42,
            alpha=0.72,
            facecolors="white",
            edgecolors=palette.get(topo, "#555555"),
            linewidths=1.1,
            label=topo,
        )

    trend = (
        d.groupby("true_time_bin", observed=True)
        .agg(
            true_time_mean=("true_time", "mean"),
            residual_mean=("residual", "mean"),
        )
        .reset_index()
        .sort_values("true_time_mean")
    )

    ax.plot(
        trend["true_time_mean"],
        trend["residual_mean"],
        color="black",
        linewidth=2.8,
        marker="o",
        markersize=5,
        label="Binned mean trend",
        zorder=8,
    )

    corr_text = "Pearson r unavailable" if corr_res is None else f"Pearson r = {corr_res:.3f}"

    ax.text(
        0.97,
        0.08,
        corr_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.92),
    )

    ax.axhline(0, linestyle="--", color="black", linewidth=1.3)
    ax.set_title("SimNet Residual vs True Completion Time")
    ax.set_xlabel("True Completion Time")
    ax.set_ylabel("Predicted Completion Time - True Completion Time")
    clean_axes(ax)
    ax.legend(fontsize=8, frameon=True, loc="upper right")
    savefig(fig, "simnet_large_02_residual_vs_true_time.png")

    # ========================================================
    # Plot 3 — Error by completion-time regime
    # ========================================================

    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    x = np.arange(len(summary_regime))
    ax.bar(
        x,
        summary_regime["abs_error_mean"],
        yerr=summary_regime["residual_std"],
        capsize=7,
        color="#8FBBD9",
        edgecolor="black",
        linewidth=0.9,
        alpha=0.85,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(summary_regime["completion_regime"], rotation=15, ha="right")
    ax.set_title("SimNet Error Across Completion-Time Regimes")
    ax.set_xlabel("Completion-Time Regime")
    ax.set_ylabel("Mean Absolute Error with Residual Std.")
    clean_axes(ax)
    savefig(fig, "simnet_large_03_error_by_completion_regime.png")

    # ========================================================
    # Plot 4 — Topology x high-scale error
    # ========================================================

    high = d[d["completion_regime"] == "High completion time"].copy()

    high_summary = (
        high.groupby("topology_type")
        .agg(
            mae=("abs_error", "mean"),
            residual_std=("residual", "std"),
            n=("true_time", "count"),
        )
        .reset_index()
        .sort_values("mae", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    ax.bar(
        high_summary["topology_type"],
        high_summary["mae"],
        yerr=high_summary["residual_std"],
        capsize=7,
        color=[palette.get(t, "#555555") for t in high_summary["topology_type"]],
        edgecolor="black",
        linewidth=0.9,
        alpha=0.85,
    )

    for i, row in high_summary.iterrows():
        ax.text(
            i,
            row["mae"],
            f"n={int(row['n'])}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title("High Completion-Time Error by Topology")
    ax.set_xlabel("Topology Type")
    ax.set_ylabel("Mean Absolute Error with Residual Std.")
    clean_axes(ax)
    savefig(fig, "simnet_large_04_high_regime_error_by_topology.png")

    print("\nSaved plots to:")
    print(PLOT_DIR)
    print("\nSaved summaries:")
    print(out_csv_1)
    print(out_csv_2)
    print(out_json)


if __name__ == "__main__":
    main()