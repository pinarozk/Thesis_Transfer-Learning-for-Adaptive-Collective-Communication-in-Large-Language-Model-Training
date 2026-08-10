"""
plot_measurement_signature.py — the figure for the "Metric Itself Was
Wrong" slide.

WHAT IT SHOWS
Left panel : the OLD gap (agent_cct / teacher_reported_time) against
             chunk size, per family. The synthetic switched families
             appear as flat horizontal lines — a constant multiplier
             across a 64x chunk-size range. That flatness is the
             fingerprint of a unit/protocol mismatch: a genuine
             behavioural failure would disperse with chunk size.
Right panel: the CORRECTED gap (agent_cct / teacher_sim_cct) on the
             same axes. The flat offsets collapse toward 1, and the
             remaining spread is real behaviour.

The two panels share a log y-axis and a reference line at 1.0, so the
"before/after" reads at a glance from the slide.

INPUT
Both CSVs produced by src/eval/cct_gap.py:
  OLD_CSV : a run that still carried the solver-reported denominator
            (e.g. cct_gap_per_graph_baseline_v4.csv)
  NEW_CSV : the sim-vs-sim run (current cct_gap_per_graph.csv)

Required columns (script checks and reports what is missing):
  topology_name, message_size_bytes, seed, gap_ratio  [OLD]
  topology_name, message_size_bytes, seed,
  gap_ratio_sim  (falls back to gap_ratio)            [NEW]

If your OLD run is gone, set OLD_CSV = None: the script then plots the
right panel alone, and optionally reconstructs the old ratio from the
new CSV when it still carries teacher_completion_time and
teacher_sim_cct (old_gap = gap_sim * teacher_sim_cct /
teacher_completion_time).

OUTPUT
  measurement_signature.png / .pdf  (300 dpi, transparent background
  off, sized for a half-slide panel)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter

# ============================================================
# CONFIG — edit these three lines
# ============================================================

EVAL_DIR = Path("src/eval")
OLD_CSV = EVAL_DIR / "cct_gap_per_graph_baseline_v5.csv"   # or None
NEW_CSV = EVAL_DIR / "cct_gap_per_graph.csv"
OUT_STEM = EVAL_DIR / "plots" / "measurement_signature"

# Families to draw, in legend order, with slide colours.
# Keys must match topology_name in the CSVs; unknown names still plot
# (grey) so nothing is silently dropped.
FAMILY_STYLE = {
    "DGX2_2_chassis":            ("DGX2 (TE-CCL)",        "#B34A2E", "o"),
    "ring_cluster":              ("ring-cluster (ILP)",   "#0E6E6D", "s"),
    "switched_dual_star_n12_r4": ("dual-star (synth.)",   "#7B5EA7", "^"),
    "switched_swclust_2x8_r4":   ("sw-cluster (synth.)",  "#C08A2E", "D"),
    "switched_two_tier_l4x4_r8": ("two-tier (synth.)",    "#3B6FB6", "v"),
}
INK, GRID, MUT = "#0F2A44", "#DCE3E8", "#5A6472"


# ============================================================
# Loading
# ============================================================

def load(path, gap_candidates):
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        print(f"[skip] {path} not found")
        return None
    df = pd.read_csv(path)

    gap_col = next((c for c in gap_candidates if c in df.columns), None)
    if gap_col is None:
        raise SystemExit(
            f"{path.name}: none of {gap_candidates} present. "
            f"Columns are: {list(df.columns)}"
        )
    need = ["topology_name", "message_size_bytes"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"{path.name}: missing columns {missing}")

    if "sim_ok" in df.columns:
        df = df[df["sim_ok"].astype(bool)]
    df = df.rename(columns={gap_col: "gap"})
    df = df[np.isfinite(df["gap"]) & (df["gap"] > 0)]
    return df


def reconstruct_old(df_new):
    """If the old run is gone, rebuild the old ratio from the new CSV."""
    need = {"teacher_sim_cct", "teacher_completion_time"}
    if not need.issubset(df_new.columns):
        return None
    d = df_new.copy()
    d = d[np.isfinite(d["teacher_completion_time"])
          & (d["teacher_completion_time"] > 0)]
    if d.empty:
        return None
    d["gap"] = d["gap"] * d["teacher_sim_cct"] / d["teacher_completion_time"]
    print("[info] old-metric panel reconstructed from the new CSV")
    return d


# ============================================================
# Plotting
# ============================================================

def df_span(ax):
    lo, hi = ax.get_ylim()
    return hi / max(lo, 1e-12)


def draw_panel(ax, df, title, subtitle):
    """One family = one line: median gap per chunk size, seeds pooled."""
    for topo, grp in df.groupby("topology_name"):
        label, colour, marker = FAMILY_STYLE.get(
            topo, (str(topo), "#9AA6B2", "o")
        )
        agg = (grp.groupby("message_size_bytes")["gap"]
                  .median().sort_index())
        if agg.empty:
            continue
        ax.plot(agg.index, agg.values, marker=marker, markersize=5,
                linewidth=1.9, color=colour, label=label, zorder=3)

    ax.axhline(1.0, color=INK, linewidth=1.0, linestyle="--",
               alpha=0.7, zorder=2)
    ax.text(0.985, 1.0, " parity ", transform=ax.get_yaxis_transform(),
            va="center", ha="right", fontsize=8, color=INK,
            bbox=dict(fc="white", ec="none", pad=1.5))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Chunk size (bytes)", fontsize=10, color=INK)
    ax.set_title(title, fontsize=13, color=INK, pad=48, loc="left",
                 fontweight="bold")
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5,
            color=MUT, va="bottom", wrap=False, linespacing=1.6)
    ax.grid(True, which="major", color=GRID, linewidth=0.7, zorder=1)
    ax.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.6,
            zorder=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUT, labelsize=9)
    span = df_span(ax)
    if span <= 12:          # narrow range: label every minor tick
        ax.yaxis.set_major_locator(
            LogLocator(base=10, subs=(1.0, 1.2, 1.5, 2.0, 3.0, 5.0)))
    else:                   # wide range: decades only
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_minor_locator(
            LogLocator(base=10, subs=tuple(np.arange(2, 10))))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))


def flatness_report(df, label):
    """Quantify the artifact: per family, max/min of the median curve.
    ~1.0 means a constant multiplier — the fingerprint."""
    print(f"\n{label}: flatness of the median curve (max/min over sizes)")
    for topo, grp in df.groupby("topology_name"):
        med = grp.groupby("message_size_bytes")["gap"].median()
        if len(med) < 2:
            continue
        ratio = med.max() / med.min()
        print(f"  {topo:32s} n_sizes={len(med):2d}  "
              f"median {med.median():8.3f}×  spread {ratio:6.2f}×")


def main():
    df_new = load(NEW_CSV, ["gap_ratio_sim", "gap_ratio"])
    if df_new is None:
        raise SystemExit(f"{NEW_CSV} is required.")

    df_old = load(OLD_CSV, ["gap_ratio", "gap_ratio_sim"])
    if df_old is None:
        df_old = reconstruct_old(df_new)

    flatness_report(df_new, "CORRECTED (sim vs sim)")
    if df_old is not None:
        flatness_report(df_old, "OLD (solver-reported denominator)")

    n_panels = 2 if df_old is not None else 1
    fig, axes = plt.subplots(
        1, n_panels, figsize=(6.5 * n_panels, 5.0), sharey=False
    )
    axes = np.atleast_1d(axes)

    if df_old is not None:
        draw_panel(
            axes[0], df_old,
            "Before  ·  gap = agent_cct / teacher_reported_time",
            "The ILP-derived families are flat — a constant multiplier.\n"
            "DGX2 instead GROWS with size — a different error signature.",
        )
        draw_panel(
            axes[1], df_new,
            "After  ·  gap_sim = agent_cct / teacher_sim_cct",
            "ILP families collapse to parity — artifact confirmed.\n"
            "DGX2 retains a residual size-dependent gap after replay "
            "correction.",
        )
    else:
        draw_panel(
            axes[0], df_new,
            "gap_sim = agent_cct / teacher_sim_cct",
            "Both sides measured in the same simulator",
        )

    if df_old is not None:
        ax = axes[1]
        ax.annotate(
            "bandwidth-dominated\nregime (≥4MB)",
            xy=(4e6, 3.7), xytext=(3e4, 3.55),
            fontsize=8.3, color=INK, ha="left", va="center",
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1,
                             shrinkA=2, shrinkB=4),
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    OUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_STEM}.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
