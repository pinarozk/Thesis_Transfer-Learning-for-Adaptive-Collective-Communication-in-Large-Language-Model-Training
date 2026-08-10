"""
Agent evaluation suite (thesis-grade).

Answers four questions the previous eval could not:

Q1  How does accuracy vary with MESSAGE SIZE?
    -> per-size-bucket load correlation (Spearman/Pearson) / MAE and
       scheduling MAE (the alpha-beta regime shifts with message size;
       a model that only works in one regime is not a CCL synthesizer).

Q2  How does accuracy vary with SWITCH INVOLVEMENT?
    -> edge-level split (is_switch_edge = 0/1) AND graph-level split
       (switch-based vs switchless topologies).

Q3  IS THE AGENT JUST PICKING SHORTEST PATHS, OR LEARNING OPTIMIZATION?
    Two independent probes:
    (a) SHORTEST-PATH DISAGREEMENT PROBE: build a shortest-path-union
        baseline (all-pairs Dijkstra by latency, + hop-count variant;
        this is what a structure-only heuristic would produce for
        AllGather). On edges where the SP baseline DISAGREES with the
        teacher's used/unused label, measure whom the agent's
        used-edge call (load_pred > LOAD_USED_THRESHOLD) matches.
        Agent ~ SP there => it learned topology heuristics. Agent ~
        teacher there => it learned something beyond shortest paths.
    (b) CAPACITY SENSITIVITY PROBE: perturb link capacities (the one
        quantity shortest paths cannot see, but the ILP optimizes
        over) and measure |delta load_pred| of the agent on perturbed
        vs untouched edges. Near-zero response => capacity-blind =>
        shortest-path-like behavior, regardless of the used-edge metric.

Q4  END-TO-END: what does an actual input/output look like?
    -> per-topology case-study figure: topology (capacities, switches),
       teacher schedule (first-use time), agent PREDICTED LOAD, and the
       decision map (TP/FP/FN/TN, from thresholding load_pred) side by
       side.

Also: per-graph CSV, JSON summary, multi-seed aggregation.

IMPORTANT (routing -> load regression, see agent_gnn.py fix #8):
routing_positive_ratio is ~0.99-1.00 across all three teachers on
these topologies, so a "used/unused" classification metric alone is
near-degenerate (a trivial always-positive baseline scores ~1.0 F1).
The PRIMARY metric here is load regression quality (Pearson/Spearman
correlation, MAE against log1p edge-load); the used-edge
precision/recall/F1 fields that remain are DIAGNOSTIC ONLY, derived by
thresholding load_pred at LOAD_USED_THRESHOLD. The old reliability
diagram / temperature calibration is GONE: there is no probability to
calibrate for a regression output (removed along with agent_gnn.py's
temperature buffer).

Compatible with: agent_gnn.py (load_head), train_agent.py
checkpoints (agent_best_seed{seed}.pt with config, no more
tuned_threshold/temperature), convert_to_pyg.py schema (y_load,
sched_mask, group_key, edge_feature_names).

Requires: networkx, pandas, matplotlib.
"""

import math
import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import networkx as nx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.models.agent_gnn import StrongAgentGNN  # noqa: E402


# ============================================================
# Paths / config
# ============================================================

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
EVAL_DIR = BASE_DIR / "src" / "eval"
PLOT_DIR = EVAL_DIR / "plots"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

TEST_PYG = DATA_DIR / "pyg_agent_test.pt"

SEEDS = [42, 1337, 2024]          # must match train_agent.py
CASE_STUDY_SEED = SEEDS[0]
CAPACITY_PERTURB_FACTOR = 0.25    # x0.25 capacity on probed edges
N_CASE_STUDIES = 4                # one per test topology, up to N

# Diagnostic-only cut for the used-edge side metric, in log1p(load)
# space -- log1p(1) == the model predicting >=1 whole chunk crossed
# the edge. Must match train_agent.py's LOAD_USED_THRESHOLD.
LOAD_USED_THRESHOLD = math.log1p(1.0)

SUMMARY_PATH = EVAL_DIR / "agent_eval_summary.json"
GRAPH_CSV_PATH = EVAL_DIR / "agent_eval_per_graph.csv"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 120,
    "savefig.dpi": 300,
})


def clean_axes(ax):
    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def savefig(fig, name):
    out = PLOT_DIR / name
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


# ============================================================
# Metrics (self-contained)
# ============================================================

def binary_metrics(scores, y, threshold):
    """DIAGNOSTIC ONLY (see module docstring). `scores` is load_pred
    (log1p-load regression output), `threshold` is in the same space
    (default LOAD_USED_THRESHOLD), NOT a [0,1] probability cut."""
    pred = (scores >= threshold).astype(float)
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    acc = (tp + tn) / max(1.0, tp + tn + fp + fn)
    return {"precision": p, "recall": r, "f1": f1, "accuracy": acc,
            "n": int(tp + tn + fp + fn)}


def regression_metrics(pred, target):
    """PRIMARY routing-side metric: load_pred vs y_load (log1p scale).
    Replaces the old binary F1/AUC/ECE trio, which were near-degenerate
    on this dataset's ~0.99-1.00 routing_positive_ratio."""
    err = pred - target
    pr = np.corrcoef(pred, target)[0, 1] if pred.std() > 0 and target.std() > 0 else float("nan")
    rp, rt = pd.Series(pred).rank().values, pd.Series(target).rank().values
    sr = (np.corrcoef(rp, rt)[0, 1]
          if rp.std() > 0 and rt.std() > 0 else float("nan"))
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "pearson_r": float(pr),
        "spearman_r": float(sr),
        "n": int(len(pred)),
    }


# ============================================================
# Checkpoint / model loading
# ============================================================

def load_model(seed, device):
    path = MODEL_DIR / f"agent_best_seed{seed}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Run train_agent.py first."
        )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = StrongAgentGNN(
        node_in_dim=cfg["node_in_dim"],
        edge_in_dim=cfg["edge_in_dim"],
        global_in_dim=cfg["global_in_dim"],
        hidden_dim=cfg["hidden_dim"],
        heads=cfg["heads"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        use_graph_context=True,
        scheduling_activation=cfg["scheduling_activation"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # no more tuned_threshold/temperature in the checkpoint -- the
    # diagnostic used-edge cut is a fixed constant in load-space, not
    # something fit on validation probabilities (see module docstring).
    # load_target_mode: "residual" (see load_priors.py) means the raw
    # model output must have the analytic sp_load baseline added back
    # to land in the same absolute log1p(load) space as y_load /
    # LOAD_USED_THRESHOLD -- done uniformly inside predict_graph.
    load_target_mode = cfg.get("load_target_mode", "absolute")
    return model, LOAD_USED_THRESHOLD, load_target_mode


# ============================================================
# Per-graph inference
# ============================================================

@torch.no_grad()
def predict_graph(model, data, device, edge_attr_override=None,
                  load_target_mode="absolute"):
    """Returns (load_pred, sched_pred) in ABSOLUTE log1p(load) space
    regardless of what the checkpoint was trained on -- callers never
    need to know about the residual/baseline split. load_target_mode
    comes from load_model()'s checkpoint config."""
    x = data.x.to(device)
    ei = data.edge_index.to(device)
    ea = (edge_attr_override if edge_attr_override is not None
          else data.edge_attr).to(device)
    u = data.u
    if u is not None and u.dim() == 1:
        u = u.unsqueeze(0)
    u = u.to(device)

    load_pred, sched = model(x, ei, ea, batch=None, u=u)
    if load_target_mode == "residual":
        load_pred = load_pred + data.sp_load_log1p.to(device)
    return load_pred.cpu().numpy(), sched.cpu().numpy()


def feat_idx(data, name):
    names = data.edge_feature_names
    if name not in names:
        raise KeyError(f"edge feature '{name}' not in {names}")
    return names.index(name)


# ============================================================
# Shortest-path baseline (Q3a)
# ============================================================

def shortest_path_union(data, weight_mode="latency"):
    """
    Structure-only heuristic for AllGather: union of shortest paths
    over all ordered (src, dst) pairs. Returns a 0/1 edge-usage vector.
    weight_mode: 'latency' (Dijkstra on latency) or 'hops'.
    """
    ei = data.edge_index.numpy()
    n_edges = ei.shape[1]
    lat_i = feat_idx(data, "latency_rel")
    lat = data.edge_attr[:, lat_i].numpy()

    G = nx.DiGraph()
    for k in range(n_edges):
        u, v = int(ei[0, k]), int(ei[1, k])
        w = 1.0 if weight_mode == "hops" else float(lat[k]) + 1e-6
        # keep the cheapest parallel edge if duplicates exist
        if G.has_edge(u, v):
            if w < G[u][v]["weight"]:
                G[u][v].update(weight=w, idx=k)
        else:
            G.add_edge(u, v, weight=w, idx=k)

    used = np.zeros(n_edges)
    for s in G.nodes:
        try:
            _, paths = nx.single_source_dijkstra(G, s, weight="weight")
        except nx.NetworkXNoPath:
            continue
        for t, path in paths.items():
            if t == s:
                continue
            for a, b in zip(path[:-1], path[1:]):
                used[G[a][b]["idx"]] = 1.0
    return used


def shortest_path_probe(per_graph, threshold):
    """
    On edges where SP baseline and teacher DISAGREE, whom does the
    agent match? Returns per-baseline stats + verdict inputs.
    """
    results = {}
    for mode in ("latency", "hops"):
        sp_all, y_all, agent_all = [], [], []
        for g in per_graph:
            sp_all.append(g[f"sp_{mode}"])
            y_all.append(g["y_routing"])
            agent_all.append((g["load_pred"] >= threshold).astype(float))
        sp = np.concatenate(sp_all)
        y = np.concatenate(y_all)
        ag = np.concatenate(agent_all)

        sp_f1 = binary_metrics(sp, y, 0.5)["f1"]
        agent_f1 = binary_metrics(ag, y, 0.5)["f1"]
        agent_sp_agreement = float((ag == sp).mean())

        dis = sp != y                      # SP is wrong here by def.
        n_dis = int(dis.sum())
        if n_dis > 0:
            agent_matches_teacher = float((ag[dis] == y[dis]).mean())
            agent_matches_sp = float((ag[dis] == sp[dis]).mean())
        else:
            agent_matches_teacher = float("nan")
            agent_matches_sp = float("nan")

        results[mode] = {
            "sp_vs_teacher_f1": sp_f1,
            "agent_vs_teacher_f1": agent_f1,
            "agent_sp_agreement": agent_sp_agreement,
            "disagreement_edges": n_dis,
            "disagreement_share": float(dis.mean()),
            "agent_matches_teacher_on_disagreement":
                agent_matches_teacher,
            "agent_matches_sp_on_disagreement": agent_matches_sp,
        }
    return results


# ============================================================
# Capacity sensitivity probe (Q3b)
# ============================================================

@torch.no_grad()
def capacity_sensitivity_probe(model, test_data, device, rng,
                               factor=CAPACITY_PERTURB_FACTOR,
                               frac_edges=0.2, load_target_mode="absolute"):
    """
    Multiply capacity by `factor` on a random subset of edges, rebuild
    the capacity-derived features consistently, and measure
    |delta load_pred|. Shortest paths (hop or latency based) are
    INVARIANT to this perturbation; the ILP optimum is not. A
    capacity-blind agent is a shortest-path learner regardless of its
    used-edge metric.
    """
    d_pert, d_ctrl = [], []

    for data in test_data:
        base_load, _ = predict_graph(model, data, device,
                                     load_target_mode=load_target_mode)

        ea = data.edge_attr.clone()
        cap_rel_i = feat_idx(data, "capacity_rel")
        log_cap_i = feat_idx(data, "log10_capacity")
        tx_i = feat_idx(data, "log10_tx_cost")

        n_edges = ea.size(0)
        n_pick = max(1, int(frac_edges * n_edges))
        picked = rng.choice(n_edges, size=n_pick, replace=False)
        mask = np.zeros(n_edges, dtype=bool)
        mask[picked] = True

        log_f = float(np.log10(factor))
        # absolute features shift exactly
        ea[picked, log_cap_i] += log_f
        ea[picked, tx_i] -= log_f
        # relative feature: recompute against (possibly new) max
        cap_abs = 10.0 ** ea[:, log_cap_i]
        ea[:, cap_rel_i] = cap_abs / cap_abs.max()

        pert_load, _ = predict_graph(model, data, device,
                                     edge_attr_override=ea,
                                     load_target_mode=load_target_mode)

        delta = np.abs(pert_load - base_load)
        d_pert.append(delta[mask])
        d_ctrl.append(delta[~mask])

    d_pert = np.concatenate(d_pert)
    d_ctrl = np.concatenate(d_ctrl)
    return {
        "factor": factor,
        "mean_abs_dload_perturbed": float(d_pert.mean()),
        "p90_abs_dload_perturbed": float(np.quantile(d_pert, 0.9)),
        "mean_abs_dload_control": float(d_ctrl.mean()),
        "sensitivity_ratio": float(
            d_pert.mean() / max(1e-8, d_ctrl.mean())
        ),
        "_raw_perturbed": d_pert,
        "_raw_control": d_ctrl,
    }


# ============================================================
# Collect everything per graph
# ============================================================

def collect(model, test_data, device, load_target_mode="absolute"):
    per_graph = []
    for idx, data in enumerate(test_data):
        load_pred, sched = predict_graph(model, data, device,
                                         load_target_mode=load_target_mode)
        y_l = data.y_load.numpy().astype(float)
        y_r = data.y_routing.numpy().astype(float)
        y_s = data.y_scheduling.numpy().astype(float)
        mask = data.sched_mask.numpy().astype(bool)
        switch_edge = data.edge_attr[
            :, feat_idx(data, "is_switch_edge")
        ].numpy() > 0.5

        per_graph.append({
            "idx": idx,
            "data": data,
            "topology": str(data.topology_name),
            "group_key": str(data.group_key),
            "source": str(data.source_domain),
            "message_bytes": float(data.message_size_bytes),
            "num_nodes": int(data.x.size(0)),
            "num_edges": int(data.edge_index.size(1)),
            "has_switch": bool(len(data.switch_indices or []) > 0),
            "load_pred": load_pred,
            "sched_pred": sched,
            "y_load": y_l,
            "y_routing": y_r,
            "y_scheduling": y_s,
            "sched_mask": mask,
            "switch_edge": switch_edge,
            "sp_latency": shortest_path_union(data, "latency"),
            "sp_hops": shortest_path_union(data, "hops"),
        })
    return per_graph


# ============================================================
# Q1: message size breakdown
# ============================================================

def message_size_analysis(per_graph, threshold):
    by_size = defaultdict(lambda: {"load_pred": [], "y_load": [],
                                   "y_routing": [],
                                   "se": [], "n_graphs": 0})
    for g in per_graph:
        b = by_size[g["message_bytes"]]
        b["load_pred"].append(g["load_pred"])
        b["y_load"].append(g["y_load"])
        b["y_routing"].append(g["y_routing"])
        m = g["sched_mask"]
        if m.any():
            b["se"].append(
                np.abs(g["sched_pred"][m] - g["y_scheduling"][m])
            )
        b["n_graphs"] += 1

    rows = []
    for size in sorted(by_size):
        b = by_size[size]
        load_pred = np.concatenate(b["load_pred"])
        y_load = np.concatenate(b["y_load"])
        y_routing = np.concatenate(b["y_routing"])
        rm = regression_metrics(load_pred, y_load)
        bm = binary_metrics(load_pred, y_routing, threshold)
        row = {"message_bytes": size,
               "message_mb": size / 1e6,
               "n_graphs": b["n_graphs"],
               **{f"load_{k}": v for k, v in rm.items()},
               **{f"used_edge_{k}": v for k, v in bm.items()}}
        if b["se"]:
            se = np.concatenate(b["se"])
            row["sched_mae"] = float(se.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_message_size(df_size):
    fig, ax1 = plt.subplots(figsize=(8.4, 5.0))
    ax1.plot(df_size["message_mb"], df_size["load_spearman_r"],
             "o-", linewidth=2, color="#2C7FB8",
             label="Load Spearman r")
    ax1.set_xscale("log")
    ax1.set_xlabel("Message size (MB, log scale)")
    ax1.set_ylabel("Load Spearman r", color="#2C7FB8")
    ax1.set_ylim(-0.05, 1.02)
    clean_axes(ax1)

    if "sched_mae" in df_size:
        ax2 = ax1.twinx()
        ax2.plot(df_size["message_mb"], df_size["sched_mae"],
                 "s--", linewidth=2, color="#F28E2B",
                 label="Scheduling MAE")
        ax2.set_ylabel("Scheduling MAE (used edges)", color="#F28E2B")
        ax2.spines["top"].set_visible(False)

    ax1.set_title("Q1 — Accuracy vs Message Size")
    savefig(fig, "q1_accuracy_vs_message_size.png")


# ============================================================
# Q2: switch involvement
# ============================================================

def switch_analysis(per_graph, threshold):
    # edge level
    edge_groups = {"switch_edges": ([], [], []),
                   "non_switch_edges": ([], [], [])}
    # graph level
    graph_groups = {"switch_topologies": ([], [], []),
                    "switchless_topologies": ([], [], [])}

    for g in per_graph:
        se = g["switch_edge"]
        edge_groups["switch_edges"][0].append(g["load_pred"][se])
        edge_groups["switch_edges"][1].append(g["y_load"][se])
        edge_groups["switch_edges"][2].append(g["y_routing"][se])
        edge_groups["non_switch_edges"][0].append(g["load_pred"][~se])
        edge_groups["non_switch_edges"][1].append(g["y_load"][~se])
        edge_groups["non_switch_edges"][2].append(g["y_routing"][~se])

        key = ("switch_topologies" if g["has_switch"]
               else "switchless_topologies")
        graph_groups[key][0].append(g["load_pred"])
        graph_groups[key][1].append(g["y_load"])
        graph_groups[key][2].append(g["y_routing"])

    out = {}
    for name, (p, yl, yr) in {**edge_groups, **graph_groups}.items():
        if not p or sum(len(a) for a in p) == 0:
            continue
        load_pred = np.concatenate(p)
        y_load = np.concatenate(yl)
        y_routing = np.concatenate(yr)
        out[name] = regression_metrics(load_pred, y_load)
        used = binary_metrics(load_pred, y_routing, threshold)
        out[name].update({f"used_edge_{k}": v for k, v in used.items()})
    return out


def plot_switch(switch_res):
    keys = [k for k in ("non_switch_edges", "switch_edges",
                        "switchless_topologies", "switch_topologies")
            if k in switch_res]
    rs = [switch_res[k]["spearman_r"] for k in keys]
    ns = [switch_res[k]["used_edge_n"] for k in keys]

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    bars = ax.bar(range(len(keys)), rs, color="#9ECAE1",
                  edgecolor="black")
    for b, n in zip(bars, ns):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"n={n}", ha="center", fontsize=9)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.replace("_", "\n") for k in keys])
    ax.set_ylabel("Load Spearman r")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Q2 — Load Prediction Quality by Switch Involvement")
    clean_axes(ax)
    savefig(fig, "q2_switch_involvement.png")


# ============================================================
# Q3 plots + verdict
# ============================================================

def plot_sp_probe(sp_res):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for ax, mode in zip(axes, ("latency", "hops")):
        r = sp_res[mode]
        labels = ["SP baseline\nvs teacher (F1)",
                  "Agent\nvs teacher (F1)",
                  "Agent matches\nTEACHER on\ndisagreement",
                  "Agent matches\nSP on\ndisagreement"]
        vals = [r["sp_vs_teacher_f1"], r["agent_vs_teacher_f1"],
                r["agent_matches_teacher_on_disagreement"],
                r["agent_matches_sp_on_disagreement"]]
        colors = ["#BBBBBB", "#2C7FB8", "#54A24B", "#E45756"]
        ax.bar(range(4), vals, color=colors, edgecolor="black")
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, ls="--", c="black", lw=1,
                   label="Coin flip on disagreement set")
        ax.set_title(f"SP weight mode: {mode} "
                     f"(disagreement edges: "
                     f"{r['disagreement_edges']})")
        ax.legend(fontsize=8)
        clean_axes(ax)
    fig.suptitle("Q3a — Shortest-Path Probe: whom does the agent copy "
                 "where SP and the optimizer disagree?")
    savefig(fig, "q3a_shortest_path_probe.png")


def plot_capacity_probe(cap_res):
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    bins = np.linspace(0, max(0.05,
                              float(cap_res["_raw_perturbed"].max())),
                       40)
    ax.hist(cap_res["_raw_control"], bins=bins, alpha=0.65,
            label="Untouched edges", color="#BBBBBB",
            edgecolor="black", linewidth=0.3, density=True)
    ax.hist(cap_res["_raw_perturbed"], bins=bins, alpha=0.65,
            label=f"Capacity x{cap_res['factor']} edges",
            color="#E45756", edgecolor="black", linewidth=0.3,
            density=True)
    ax.set_xlabel("|Δ predicted load (log1p)| after capacity perturbation")
    ax.set_ylabel("Density")
    ax.set_title(
        "Q3b — Capacity Sensitivity "
        f"(ratio: {cap_res['sensitivity_ratio']:.1f}x). "
        "Shortest paths are invariant to this perturbation."
    )
    ax.legend()
    clean_axes(ax)
    savefig(fig, "q3b_capacity_sensitivity.png")


def sp_verdict(sp_res, cap_res):
    """Plain-language verdict for the thesis discussion section."""
    r = sp_res["latency"]
    lines = []
    amt = r["agent_matches_teacher_on_disagreement"]
    ratio = cap_res["sensitivity_ratio"]

    if amt == amt:  # not NaN
        if amt > 0.6:
            lines.append(
                f"On the {r['disagreement_edges']} edges where the "
                f"shortest-path heuristic contradicts the teacher, the "
                f"agent sides with the TEACHER {amt:.0%} of the time "
                f"(coin flip = 50%). This is direct evidence of "
                f"learning beyond shortest-path structure.")
        elif amt < 0.45:
            lines.append(
                f"On SP-vs-teacher disagreement edges the agent sides "
                f"with the SP heuristic ({1-amt:.0%}). The agent is "
                f"largely reproducing shortest-path structure — "
                f"discuss as a limitation.")
        else:
            lines.append(
                f"On disagreement edges the agent is near chance "
                f"({amt:.0%}) — inconclusive; more data or capacity-"
                f"aware features may be needed.")

    if ratio > 3:
        lines.append(
            f"Capacity probe: perturbed edges respond {ratio:.1f}x "
            f"more than controls — the agent USES capacity, which "
            f"shortest paths cannot.")
    elif ratio < 1.5:
        lines.append(
            f"Capacity probe: response ratio only {ratio:.1f}x — the "
            f"agent is nearly capacity-blind, consistent with "
            f"shortest-path-like behavior.")
    else:
        lines.append(
            f"Capacity probe: moderate sensitivity ({ratio:.1f}x).")
    return lines


# ============================================================
# Load regression diagnostic plot + used-edge threshold sweep
# ============================================================
# The old reliability diagram / BCE calibration is GONE: there is no
# probability to calibrate for a regression output (agent_gnn.py's
# temperature buffer was removed along with the binary routing head).
# plot_load_scatter replaces it as the natural regression diagnostic;
# plot_threshold_sweep is kept but reframed as diagnostic-only (it
# sweeps LOAD_USED_THRESHOLD, not a probability cut, and answers a
# downstream-decoding question -- "how aggressively could
# schedule_bridge.py prune on load_pred" -- not model quality).

def plot_load_scatter(per_graph):
    load_pred = np.concatenate([g["load_pred"] for g in per_graph])
    y_load = np.concatenate([g["y_load"] for g in per_graph])
    rm = regression_metrics(load_pred, y_load)

    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    lo = min(load_pred.min(), y_load.min())
    hi = max(load_pred.max(), y_load.max())
    ax.plot([lo, hi], [lo, hi], "--", c="black", lw=1.4,
            label="Perfect prediction")
    ax.scatter(y_load, load_pred, s=10, alpha=0.35, color="#2C7FB8")
    ax.set_xlabel("Teacher log1p(edge load)")
    ax.set_ylabel("Predicted log1p(edge load)")
    ax.set_title(f"Load Prediction (Pearson r={rm['pearson_r']:.3f}, "
                f"Spearman r={rm['spearman_r']:.3f}, "
                f"MAE={rm['mae']:.3f})")
    ax.legend()
    clean_axes(ax)
    savefig(fig, "load_prediction_scatter.png")
    return rm


def plot_threshold_sweep(per_graph):
    load_pred = np.concatenate([g["load_pred"] for g in per_graph])
    y_routing = np.concatenate([g["y_routing"] for g in per_graph])
    lo, hi = float(load_pred.min()), float(load_pred.max())
    ts = np.linspace(max(0.0, lo), max(0.01, hi), 19)
    ps, rs, f1s = [], [], []
    for t in ts:
        m = binary_metrics(load_pred, y_routing, t)
        ps.append(m["precision"]); rs.append(m["recall"])
        f1s.append(m["f1"])

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.plot(ts, ps, "-o", label="Precision", color="#2C7FB8")
    ax.plot(ts, rs, "-s", label="Recall", color="#F28E2B")
    ax.plot(ts, f1s, "-^", label="F1", color="#54A24B")
    ax.axvline(LOAD_USED_THRESHOLD, ls="--", c="gray", lw=1,
              label="Default cut (log1p(1))")
    ax.set_xlabel("load_pred threshold (log1p space)")
    ax.set_ylabel("Score (used-edge diagnostic)")
    ax.set_title("DIAGNOSTIC: used-edge threshold sweep (test) -- "
                "not the primary load-regression metric")
    ax.legend()
    clean_axes(ax)
    savefig(fig, "threshold_sweep.png")


# ============================================================
# Q4: end-to-end case studies
# ============================================================

def _layout(data):
    G = nx.Graph()
    ei = data.edge_index.numpy()
    for k in range(ei.shape[1]):
        G.add_edge(int(ei[0, k]), int(ei[1, k]))
    return G, nx.kamada_kawai_layout(G)


def _draw_nodes(ax, pos, data):
    switches = set(data.switch_indices or [])
    gpus = [n for n in pos if n not in switches]
    xs = [pos[n][0] for n in gpus]; ys = [pos[n][1] for n in gpus]
    ax.scatter(xs, ys, s=340, c="#EAF2F8", edgecolors="black",
               zorder=3)
    if switches:
        xs = [pos[n][0] for n in switches]
        ys = [pos[n][1] for n in switches]
        ax.scatter(xs, ys, s=380, c="#FDD0A2", edgecolors="black",
                   marker="s", zorder=3, label="switch")
    for n, (x, y) in pos.items():
        ax.text(x, y, str(n), ha="center", va="center", fontsize=8,
                zorder=4)
    ax.set_axis_off()


def _draw_edges(ax, pos, data, colors, widths, cmap=None,
                vmin=0.0, vmax=1.0):
    ei = data.edge_index.numpy()
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        x1, y1 = pos[u]; x2, y2 = pos[v]
        c = colors[k]
        if cmap is not None:
            c = plt.get_cmap(cmap)(
                (float(colors[k]) - vmin) / max(1e-8, vmax - vmin))
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=c,
                            lw=widths[k],
                            connectionstyle="arc3,rad=0.12",
                            shrinkA=12, shrinkB=12),
            zorder=2,
        )


def case_study(g, threshold, tag):
    data = g["data"]
    G, pos = _layout(data)
    ne = data.edge_index.size(1)
    cap = data.edge_attr[:, feat_idx(data, "capacity_rel")].numpy()
    widths_cap = 0.6 + 2.4 * cap

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.6))

    # (1) topology
    ax = axes[0]
    _draw_edges(ax, pos, data, colors=["#888888"] * ne,
                widths=widths_cap)
    _draw_nodes(ax, pos, data)
    ax.set_title(f"Topology: {g['topology']}\n"
                 f"({g['num_nodes']} nodes, "
                 f"{g['message_bytes']/1e6:.2g} MB)\n"
                 f"edge width ∝ capacity")

    # (2) teacher schedule (first-use time on used edges)
    ax = axes[1]
    y = g["y_routing"]; ys = g["y_scheduling"]
    cols, wids = [], []
    for k in range(ne):
        if y[k] > 0:
            cols.append(plt.get_cmap("viridis")(ys[k]))
            wids.append(2.2)
        else:
            cols.append("#DDDDDD"); wids.append(0.5)
    _draw_edges(ax, pos, data, colors=cols, widths=wids)
    _draw_nodes(ax, pos, data)
    ax.set_title("TEACHER (ILP) schedule\ncolor = normalized "
                 "first-use time (viridis)")

    # (3) agent predicted load (min-max normalized per graph for
    # color/width only -- the CSV/summary keep the real log1p values)
    ax = axes[2]
    load_pred = g["load_pred"]
    lo, hi = float(load_pred.min()), float(load_pred.max())
    load_norm = ((load_pred - lo) / (hi - lo) if hi > lo
                else np.zeros_like(load_pred))
    cols = [plt.get_cmap("viridis")(p) for p in load_norm]
    wids = [0.5 + 2.0 * p for p in load_norm]
    _draw_edges(ax, pos, data, colors=cols, widths=wids)
    _draw_nodes(ax, pos, data)
    ax.set_title("AGENT predicted load\n(color+width = min-max "
                 "normalized log1p(load) per graph)")

    # (4) decision map (diagnostic: thresholding load_pred)
    ax = axes[3]
    pred = (load_pred >= threshold).astype(float)
    cols, wids = [], []
    for k in range(ne):
        if pred[k] == 1 and y[k] == 1:
            cols.append("#54A24B"); wids.append(2.4)   # TP
        elif pred[k] == 1 and y[k] == 0:
            cols.append("#F28E2B"); wids.append(2.0)   # FP
        elif pred[k] == 0 and y[k] == 1:
            cols.append("#E45756"); wids.append(2.0)   # FN
        else:
            cols.append("#E5E5E5"); wids.append(0.4)   # TN
    _draw_edges(ax, pos, data, colors=cols, widths=wids)
    _draw_nodes(ax, pos, data)
    m = binary_metrics(load_pred, y, threshold)
    ax.set_title(f"Used-edge decision map (diagnostic) @ "
                f"t={threshold:.2f}\n"
                f"P={m['precision']:.2f} R={m['recall']:.2f} "
                f"F1={m['f1']:.2f}")
    ax.legend(handles=[
        Line2D([0], [0], color="#54A24B", lw=2.4, label="TP"),
        Line2D([0], [0], color="#F28E2B", lw=2.0, label="FP"),
        Line2D([0], [0], color="#E45756", lw=2.0, label="FN"),
        Line2D([0], [0], color="#E5E5E5", lw=1.0, label="TN"),
    ], loc="lower left", fontsize=8)

    savefig(fig, f"q4_case_study_{tag}.png")


def make_case_studies(per_graph, threshold):
    """One representative graph per test topology (median message
    size), up to N_CASE_STUDIES."""
    by_topo = defaultdict(list)
    for g in per_graph:
        by_topo[g["topology"]].append(g)

    for i, (topo, gs) in enumerate(sorted(by_topo.items())):
        if i >= N_CASE_STUDIES:
            break
        gs = sorted(gs, key=lambda g: g["message_bytes"])
        g = gs[len(gs) // 2]
        case_study(g, threshold,
                   tag=f"{topo}_{g['message_bytes']/1e6:.2g}MB")


# ============================================================
# Per-graph CSV
# ============================================================

def per_graph_csv(per_graph, threshold):
    rows = []
    for g in per_graph:
        rm = regression_metrics(g["load_pred"], g["y_load"])
        um = binary_metrics(g["load_pred"], g["y_routing"], threshold)
        row = {
            "idx": g["idx"], "topology": g["topology"],
            "group_key": g["group_key"], "source": g["source"],
            "message_mb": g["message_bytes"] / 1e6,
            "num_nodes": g["num_nodes"], "num_edges": g["num_edges"],
            "has_switch": g["has_switch"],
            **{f"load_{k}": v for k, v in rm.items()},
            **{f"used_edge_{k}": v for k, v in um.items()},
        }
        msk = g["sched_mask"]
        if msk.any():
            row["sched_mae"] = float(np.abs(
                g["sched_pred"][msk] - g["y_scheduling"][msk]
            ).mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(GRAPH_CSV_PATH, index=False)
    print("Saved:", GRAPH_CSV_PATH)
    return df


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    test_data = torch.load(TEST_PYG, map_location="cpu",
                           weights_only=False)
    print("Test graphs:", len(test_data))

    rng = np.random.default_rng(0)
    all_seed_summaries = {}

    for seed in SEEDS:
        try:
            model, threshold, load_target_mode = load_model(seed, device)
        except FileNotFoundError as e:
            print("SKIP seed", seed, "-", e)
            continue

        print(f"\n===== seed {seed} (used-edge diagnostic cut "
              f"{threshold:.4f}, load_target_mode={load_target_mode}) "
              f"=====")
        per_graph = collect(model, test_data, device,
                            load_target_mode=load_target_mode)

        load_pred = np.concatenate([g["load_pred"] for g in per_graph])
        y_load = np.concatenate([g["y_load"] for g in per_graph])
        y_routing = np.concatenate([g["y_routing"] for g in per_graph])

        summary = {
            "threshold": threshold,
            "load_overall": regression_metrics(load_pred, y_load),
            "used_edge_overall": binary_metrics(
                load_pred, y_routing, threshold
            ),
            "message_size": message_size_analysis(
                per_graph, threshold
            ).to_dict(orient="records"),
            "switch": switch_analysis(per_graph, threshold),
            "sp_probe": shortest_path_probe(per_graph, threshold),
        }

        cap = capacity_sensitivity_probe(model, test_data, device, rng,
                                         load_target_mode=load_target_mode)
        summary["capacity_probe"] = {
            k: v for k, v in cap.items() if not k.startswith("_")
        }
        summary["verdict"] = sp_verdict(summary["sp_probe"], cap)

        all_seed_summaries[seed] = summary

        # plots + CSV + case studies only for the designated seed
        if seed == CASE_STUDY_SEED:
            df_size = message_size_analysis(per_graph, threshold)
            plot_message_size(df_size)
            plot_switch(summary["switch"])
            plot_sp_probe(summary["sp_probe"])
            plot_capacity_probe(cap)
            plot_load_scatter(per_graph)
            plot_threshold_sweep(per_graph)
            make_case_studies(per_graph, threshold)
            per_graph_csv(per_graph, threshold)

        print("\nVERDICT (seed", seed, "):")
        for line in summary["verdict"]:
            print("  •", line)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(all_seed_summaries, f, indent=2, default=float)
    print("\nSaved:", SUMMARY_PATH)
    print("Plots in:", PLOT_DIR)


if __name__ == "__main__":
    main()
