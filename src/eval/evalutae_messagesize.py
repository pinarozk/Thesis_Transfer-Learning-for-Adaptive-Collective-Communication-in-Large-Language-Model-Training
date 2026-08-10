# src/eval/analyze_simnet_message_and_topology.py

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
EVAL_DIR = BASE_DIR / "src" / "eval"
PLOT_DIR = EVAL_DIR / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

CSV_CANDIDATES = [
    EVAL_DIR / "simnet_graph_eval_rows.csv",
    EVAL_DIR / "simnet_eval_rows.csv",
    EVAL_DIR / "simnet_test_predictions.csv",
]


# ============================================================
# Style
# ============================================================

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


def find_existing_csv():
    for p in CSV_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No SimNet eval CSV found. Expected one of:\n"
        + "\n".join(str(p) for p in CSV_CANDIDATES)
    )


def pick_col(df, options, required=True):
    for c in options:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of these columns found: {options}\nAvailable columns: {list(df.columns)}")
    return None


def safe_pearson_logx(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return None

    log_x = np.log10(x)

    if np.std(log_x) == 0 or np.std(y) == 0:
        return None

    return float(np.corrcoef(log_x, y)[0, 1])


def topology_color_map(topologies):
    palette = {
        "line": "#4C78A8",
        "ring": "#F58518",
        "star": "#54A24B",
        "bottleneck": "#E45756",
        "random": "#72B7B2",
        "unknown": "#555555",
    }
    return {t: palette.get(t, "#555555") for t in topologies}


# ============================================================
# Load and normalize
# ============================================================

csv_path = find_existing_csv()
df = pd.read_csv(csv_path)

print("Loaded:", csv_path)
print("Columns:", list(df.columns))

true_col = pick_col(df, ["true_time", "completion_time", "true_completion_time", "y_true", "true"])
pred_col = pick_col(df, ["pred_time", "pred_completion_time", "prediction", "y_pred", "pred"])
topo_col = pick_col(df, ["topology_type", "topology_group", "topology_name"])
msg_col = pick_col(df, ["chunk_mb", "message_mb", "message_size_mb", "chunk_size_mb"])

d = df.copy()

d["true_time"] = d[true_col].astype(float)
d["pred_time"] = d[pred_col].astype(float)
d["topology_type"] = d[topo_col].astype(str)
d["message_mb"] = d[msg_col].astype(float)

d = d.replace([np.inf, -np.inf], np.nan)
d = d.dropna(subset=["true_time", "pred_time", "topology_type", "message_mb"])
d = d[d["message_mb"] > 0].copy()

d["residual"] = d["pred_time"] - d["true_time"]
d["abs_error"] = np.abs(d["residual"])
d["squared_error"] = d["residual"] ** 2

# Avoid exploding percentage for very tiny completion times
eps = 1e-12
d["ape"] = d["abs_error"] / np.maximum(np.abs(d["true_time"]), eps)

# Since current SimNet generator has no switch nodes, use topology structure proxy.
d["topology_structure"] = np.where(
    d["topology_type"].isin(["line", "ring", "star"]),
    "Structured / Regular",
    "Irregular or Bottleneck",
)


# ============================================================
# Summary metrics
# ============================================================

def rmse(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))

summary_by_topology = (
    d.groupby("topology_type")
    .agg(
        n=("true_time", "count"),
        mean_true_time=("true_time", "mean"),
        mean_pred_time=("pred_time", "mean"),
        mean_residual=("residual", "mean"),
        std_residual=("residual", "std"),
        mae=("abs_error", "mean"),
        rmse=("residual", rmse),
        mape=("ape", "mean"),
    )
    .reset_index()
)

summary_by_structure = (
    d.groupby("topology_structure")
    .agg(
        n=("true_time", "count"),
        mean_true_time=("true_time", "mean"),
        mean_pred_time=("pred_time", "mean"),
        mean_residual=("residual", "mean"),
        std_residual=("residual", "std"),
        mae=("abs_error", "mean"),
        rmse=("residual", rmse),
        mape=("ape", "mean"),
    )
    .reset_index()
)

message_corr_residual = safe_pearson_logx(d["message_mb"], d["residual"])
message_corr_abs_error = safe_pearson_logx(d["message_mb"], d["abs_error"])

summary = {
    "csv_path": str(csv_path),
    "num_samples": int(len(d)),
    "message_size_effect": {
        "pearson_r_log_message_vs_residual": message_corr_residual,
        "pearson_r_log_message_vs_abs_error": message_corr_abs_error,
    },
    "overall": {
        "mae": float(d["abs_error"].mean()),
        "rmse": rmse(d["residual"]),
        "mean_residual": float(d["residual"].mean()),
        "std_residual": float(d["residual"].std()),
    },
}

summary_by_topology.to_csv(EVAL_DIR / "simnet_summary_by_topology.csv", index=False)
summary_by_structure.to_csv(EVAL_DIR / "simnet_summary_by_topology_structure.csv", index=False)

with open(EVAL_DIR / "simnet_message_topology_analysis_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nSummary by topology:")
print(summary_by_topology)

print("\nSummary by topology structure:")
print(summary_by_structure)

print("\nCorrelation summary:")
print(json.dumps(summary, indent=2))


# ============================================================
# Plot 1 — SimNet residual vs message size
# ============================================================

topologies = sorted(d["topology_type"].unique())
colors = topology_color_map(topologies)

fig, ax = plt.subplots(figsize=(8.7, 5.3))

for topo in topologies:
    sub = d[d["topology_type"] == topo]

    ax.scatter(
        sub["message_mb"],
        sub["residual"],
        s=42,
        alpha=0.72,
        facecolors="white",
        edgecolors=colors[topo],
        linewidths=1.15,
        label=topo,
    )

    topo_trend = (
        sub.groupby("message_mb")["residual"]
        .mean()
        .reset_index()
        .sort_values("message_mb")
    )

    if len(topo_trend) >= 2:
        ax.plot(
            topo_trend["message_mb"],
            topo_trend["residual"],
            color=colors[topo],
            linewidth=1.8,
            alpha=0.85,
            marker="o",
            markersize=4,
        )

global_trend = (
    d.groupby("message_mb")["residual"]
    .mean()
    .reset_index()
    .sort_values("message_mb")
)

if len(global_trend) >= 2:
    ax.plot(
        global_trend["message_mb"],
        global_trend["residual"],
        color="black",
        linewidth=2.8,
        marker="o",
        markersize=5,
        label="Overall mean trend",
        zorder=8,
    )

corr_text = (
    "Pearson r unavailable"
    if message_corr_residual is None
    else f"Pearson r = {message_corr_residual:.3f}\nusing log10(message size)"
)

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
ax.set_xscale("log")
ax.set_title("SimNet Residual vs Message Size")
ax.set_xlabel("Message Size / Chunk Size (MB, log scale)")
ax.set_ylabel("Predicted Completion Time - True Completion Time")
clean_axes(ax)
ax.legend(fontsize=8, frameon=True, loc="upper right")
savefig(fig, "simnet_01_residual_vs_message_size_with_trend.png")


# ============================================================
# Plot 2 — SimNet absolute error vs message size
# ============================================================

fig, ax = plt.subplots(figsize=(8.7, 5.3))

for topo in topologies:
    sub = d[d["topology_type"] == topo]

    ax.scatter(
        sub["message_mb"],
        sub["abs_error"],
        s=42,
        alpha=0.72,
        facecolors="white",
        edgecolors=colors[topo],
        linewidths=1.15,
        label=topo,
    )

    topo_trend = (
        sub.groupby("message_mb")["abs_error"]
        .mean()
        .reset_index()
        .sort_values("message_mb")
    )

    if len(topo_trend) >= 2:
        ax.plot(
            topo_trend["message_mb"],
            topo_trend["abs_error"],
            color=colors[topo],
            linewidth=1.8,
            alpha=0.85,
            marker="o",
            markersize=4,
        )

global_trend = (
    d.groupby("message_mb")["abs_error"]
    .mean()
    .reset_index()
    .sort_values("message_mb")
)

if len(global_trend) >= 2:
    ax.plot(
        global_trend["message_mb"],
        global_trend["abs_error"],
        color="black",
        linewidth=2.8,
        marker="o",
        markersize=5,
        label="Overall mean trend",
        zorder=8,
    )

corr_text = (
    "Pearson r unavailable"
    if message_corr_abs_error is None
    else f"Pearson r = {message_corr_abs_error:.3f}\nusing log10(message size)"
)

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

ax.set_xscale("log")
ax.set_title("SimNet Absolute Error vs Message Size")
ax.set_xlabel("Message Size / Chunk Size (MB, log scale)")
ax.set_ylabel("Absolute Completion Time Error")
clean_axes(ax)
ax.legend(fontsize=8, frameon=True, loc="upper right")
savefig(fig, "simnet_02_abs_error_vs_message_size_with_trend.png")


# ============================================================
# Plot 3 — Structure proxy comparison
# ============================================================

fig, ax = plt.subplots(figsize=(7.4, 5.2))

order = ["Structured / Regular", "Irregular or Bottleneck"]
plot_df = summary_by_structure.set_index("topology_structure").reindex(order).dropna().reset_index()

ax.bar(
    plot_df["topology_structure"],
    plot_df["mae"],
    yerr=plot_df["std_residual"],
    capsize=7,
    color="#8FBBD9",
    edgecolor="black",
    linewidth=0.9,
    alpha=0.85,
)

ax.set_title("SimNet Error by Topology Structure")
ax.set_xlabel("Topology Structure")
ax.set_ylabel("MAE with Residual Std.")
clean_axes(ax)
savefig(fig, "simnet_03_error_by_topology_structure.png")


# ============================================================
# Plot 4 — Topology-wise residual boxplot
# ============================================================

labels = sorted(d["topology_type"].unique())
values = [d[d["topology_type"] == label]["residual"].values for label in labels]

fig, ax = plt.subplots(figsize=(8.7, 5.3))

box = ax.boxplot(
    values,
    labels=labels,
    patch_artist=True,
    showfliers=True,
    medianprops=dict(color="black", linewidth=1.4),
    boxprops=dict(edgecolor="black", linewidth=1.0),
    whiskerprops=dict(color="black", linewidth=1.0),
    capprops=dict(color="black", linewidth=1.0),
)

for patch, label in zip(box["boxes"], labels):
    patch.set_facecolor(colors[label])
    patch.set_alpha(0.45)

ax.axhline(0, linestyle="--", color="black", linewidth=1.3)
ax.set_title("SimNet Residual by Topology Type")
ax.set_xlabel("Topology Type")
ax.set_ylabel("Predicted Completion Time - True Completion Time")
clean_axes(ax)
savefig(fig, "simnet_04_residual_by_topology_type.png")


print("\nSaved SimNet analysis plots to:")
print(PLOT_DIR)