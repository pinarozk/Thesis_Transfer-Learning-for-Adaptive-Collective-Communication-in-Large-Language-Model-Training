"""
worked_example_figures.py — the two diagrams for the worked-example
slide, generated from real data/checkpoints (nothing hand-drawn).

Figure A (topology): reads the REAL edge_index of the picked test
instance from pyg_agent_test.pt and draws it with a networkx spring
layout. Source nodes (is_source=1) are white-fill/navy-outline circles;
relay/switch nodes are solid orange — same visual language as the
topology gallery slide. The busiest edge (max of y_load, the ground-
truth edge-load target) is drawn as a thick red-orange line; its
upstream node is marked with an arrow.

Figure B (decoded tree): runs the ACTUAL frozen Agent checkpoint and
calls decode_allgather for real, then draws source node 0's own
sub-tree (its "S0_"-prefixed policy entries) as a layered BFS-depth
tree — the exact visual counterpart of the "first 3 transfers" table
on the worked-example slide, but complete and to scale.

Selection rule: IDENTICAL to worked_example.py, on purpose, so both
scripts pick the same graph when given the same --topology/--size.
That is: within the test graphs for --topology, pick the one whose
message_size_bytes is closest to --size (default: the median of the
available sizes). Cross-check via the "Busiest edge" line this script
prints at the end against worked_example.py's [2] EDGE line — they
must name the same (u, v).

Usage:
    python worked_example_figures.py
    python worked_example_figures.py --topology ring_cluster --size 2000000

Output (300 dpi, PNG + SVG, pptx-ready):
    src/eval/plots/worked_example_topology.png / .svg
    src/eval/plots/worked_example_tree.png / .svg
"""

import argparse
import sys
from collections import defaultdict, deque
from pathlib import Path

import networkx as nx
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "CCL_Simulator"))

from src.models.agent_gnn import StrongAgentGNN            # noqa: E402
from src.pipeline.schedule_bridge import decode_allgather  # noqa: E402

DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
OUT_DIR = ROOT / "src" / "eval" / "plots"
SEED = 42

INK = "#0f2a44"
INK_DIM = "#4d5b68"
SOURCE_EDGE = "#0f2a44"
RELAY_FILL = "#c9781f"
BUSIEST = "#c0392b"
TREE_ACCENT = "#0e6e6d"
LINE = "#c7cdcb"


# ============================================================
# Instance selection — must mirror worked_example.py exactly
# ============================================================

def pick_instance(pyg, topology, size):
    cand = [d for d in pyg if str(d.topology_name) == topology]
    if not cand:
        raise SystemExit(f"No test graphs named {topology!r}.")
    sizes = sorted({float(d.message_size_bytes) for d in cand})
    target = size if size is not None else sizes[len(sizes) // 2]
    data = min(cand, key=lambda d: abs(float(d.message_size_bytes) - target))
    return data


def load_agent(device):
    ck = torch.load(MODELS / f"agent_best_seed{SEED}.pt",
                    map_location=device, weights_only=False)
    cfg = ck["config"]
    m = StrongAgentGNN(
        node_in_dim=cfg["node_in_dim"], edge_in_dim=cfg["edge_in_dim"],
        global_in_dim=cfg["global_in_dim"], hidden_dim=cfg["hidden_dim"],
        heads=cfg["heads"], num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        scheduling_activation=cfg["scheduling_activation"],
    ).to(device)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m


# ============================================================
# Figure A — real topology
#
# Preferred layout: TWO SQUARES SIDE BY SIDE, same visual language as
# the topology gallery's two_bridge scheme -- not decorative, it is
# the actual proof of "two rings + bridges", not just a claim in a
# caption. Detected generically (Kernighan-Lin min-cut bisection into
# two equal clusters, each drawn in cycle order if it happens to BE a
# cycle); falls back to a spring layout for topologies that don't
# decompose this way (e.g. hub-and-spoke families).
# ============================================================

def _cycle_order(nodes, subG):
    """If subG (induced on `nodes`) is a single simple cycle, return
    the nodes in cycle order starting from the smallest id. Else None."""
    if any(subG.degree(n) != 2 for n in nodes):
        return None
    start = min(nodes)
    order = [start]
    prev, cur = None, start
    while len(order) < len(nodes):
        nxts = [x for x in subG.neighbors(cur) if x != prev]
        if not nxts:
            return None
        prev, cur = cur, nxts[0]
        if cur in order:
            break
        order.append(cur)
    return order if len(order) == len(nodes) else None


def try_two_square_layout(G):
    """Bisect into two equal clusters joined by bridges; place each
    cluster as a diamond ('square') of its own cycle order. Returns
    None if the graph doesn't cleanly fit this shape (uneven halves,
    or a side that isn't itself a simple cycle)."""
    n = G.number_of_nodes()
    if n < 4 or n % 2 != 0:
        return None
    try:
        part_a, part_b = nx.algorithms.community.kernighan_lin_bisection(
            G, max_iter=50, seed=SEED)
    except Exception:
        return None
    if len(part_a) != len(part_b):
        return None

    order_a = _cycle_order(sorted(part_a), G.subgraph(part_a))
    order_b = _cycle_order(sorted(part_b), G.subgraph(part_b))
    if order_a is None or order_b is None:
        return None

    def diamond(order, cx):
        slots = [(0, 0.8), (0.8, 0.0), (0, -0.8), (-0.8, 0.0)]
        pos = {}
        for i, node in enumerate(order):
            dx, dy = slots[i % 4]
            pos[node] = (cx + dx, dy)
        return pos

    pos = {}
    pos.update(diamond(order_a, -1.5))
    pos.update(diamond(order_b, 1.5))
    return pos


def draw_topology(data, out_stem):
    ei = data.edge_index.numpy()
    n = int(data.x.size(0))
    node_names = list(data.node_feature_names)
    src_i = node_names.index("is_source")
    is_source = (data.x[:, src_i].numpy() > 0.5)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        G.add_edge(u, v)

    y = data.y_load.view(-1).numpy()
    k_busiest = int(np.argmax(y))
    bu, bv = int(ei[0, k_busiest]), int(ei[1, k_busiest])

    pos = try_two_square_layout(G)
    two_square = pos is not None
    if pos is None:
        pos = nx.spring_layout(G, seed=SEED, k=1.4 / max(np.sqrt(n), 1))

    fig, ax = plt.subplots(figsize=(6.4, 4.8), facecolor="#ffffff")
    ax.axis("off")
    ax.set_aspect("equal")

    for u, v in G.edges():
        if {u, v} == {bu, bv}:
            continue
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        is_bridge = two_square and (pos[u][0] < 0) != (pos[v][0] < 0)
        if is_bridge:
            ax.plot([x1, x2], [y1, y2], color=INK_DIM, lw=1.5, ls="--",
                     alpha=0.85, zorder=1)
            continue
        ax.plot([x1, x2], [y1, y2], color=LINE, lw=1.6, zorder=1)

    x1, y1 = pos[bu]
    x2, y2 = pos[bv]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=BUSIEST, lw=3.2,
                                 shrinkA=14, shrinkB=14, mutation_scale=18),
                zorder=4)

    for node in G.nodes():
        x, y = pos[node]
        if is_source[node]:
            ax.add_patch(Circle((x, y), 0.045, facecolor="#ffffff",
                                 edgecolor=SOURCE_EDGE, lw=1.6, zorder=3))
            tcolor = INK
        else:
            ax.add_patch(Circle((x, y), 0.045, facecolor=RELAY_FILL,
                                 edgecolor=INK, lw=1.2, zorder=3))
            tcolor = "#ffffff"
        ax.text(x, y, str(node), ha="center", va="center", fontsize=8.5,
                fontweight="bold", color=tcolor, zorder=5)
        if node == 0:
            ax.annotate("node 0", xy=(x, y), xytext=(x + 0.10, y + 0.14),
                        fontsize=8.5, color=INK_DIM,
                        arrowprops=dict(arrowstyle="-", color=INK_DIM, lw=0.9))

    if two_square:
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-1.35, 1.25)
        left_ids = sorted(n for n in G.nodes() if pos[n][0] < 0)
        right_ids = sorted(n for n in G.nodes() if pos[n][0] > 0)
        ax.text(-1.5, 1.05, f"ring {{{','.join(map(str, left_ids))}}}",
                ha="center", fontsize=8.5, color=INK_DIM, style="italic")
        ax.text(1.5, 1.05, f"ring {{{','.join(map(str, right_ids))}}}",
                ha="center", fontsize=8.5, color=INK_DIM, style="italic")

    n_src = int(is_source.sum())
    ax.text(0.5, -0.10 if two_square else -0.06,
            f"{data.topology_name}  ·  {n_src} source / {n - n_src} relay "
            f"·  busiest edge {bu}→{bv} (thick)",
            transform=ax.transAxes, ha="center", fontsize=8.5, color=INK_DIM)

    legend = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#fff",
                    markeredgecolor=SOURCE_EDGE, markersize=9, label="source (is_source=1)"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=RELAY_FILL,
                    markeredgecolor=INK, markersize=9, label="relay / switch"),
        plt.Line2D([0], [0], color=BUSIEST, lw=3, label="busiest edge (max y_load)"),
    ]
    if two_square:
        legend.insert(2, plt.Line2D([0], [0], color=INK_DIM, lw=1.5, ls="--",
                                     label="bridge (inter-cluster)"))
    ax.legend(handles=legend, loc="lower center",
              bbox_to_anchor=(0.5, -0.22 if two_square else -0.16),
              ncol=2 if two_square else 3, frameon=False, fontsize=8)

    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="#ffffff")
    plt.close(fig)
    return bu, bv


# ============================================================
# Figure B — decoded tree from source 0, real Agent + decode_allgather
# ============================================================

def draw_tree(data, out_stem, device):
    agent = load_agent(device)
    u_vec = data.u if data.u.dim() == 2 else data.u.unsqueeze(0)
    with torch.no_grad():
        load_pred, sched_pred = agent(data.x, data.edge_index,
                                      data.edge_attr, None, u_vec)
    policy, stats = decode_allgather(data, load_pred.view(-1).numpy(),
                                     sched_pred.view(-1).numpy())

    tree = [e for e in policy if e.chunk_id.startswith("S0_")]
    edges = []
    for e in tree:
        u = int(e.src[3:]) if e.src.startswith("GPU") else int(e.src)
        v = int(e.dst[3:]) if e.dst.startswith("GPU") else int(e.dst)
        edges.append((e.chunk_id, u, v))

    children = defaultdict(list)
    for cid, u, v in edges:
        children[u].append((v, cid))

    # BFS layering from root 0
    depth = {0: 0}
    order = {0: 0}
    q = deque([0])
    layer_count = defaultdict(int)
    while q:
        u = q.popleft()
        for v, cid in sorted(children[u]):
            depth[v] = depth[u] + 1
            order[v] = layer_count[depth[v]]
            layer_count[depth[v]] += 1
            q.append(v)

    max_depth = max(depth.values()) if depth else 0
    layer_width = {d: max(layer_count[d], 1) for d in range(max_depth + 1)}
    pos = {}
    for node, d in depth.items():
        w = layer_width[d]
        x = (order[node] - (w - 1) / 2.0) * max(2.6 / max(w, 1), 1.15)
        pos[node] = (x, -d)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_pad, y_pad = 0.7, 0.6
    left_pad = 1.15  # extra room for "hop N" row labels

    fig, ax = plt.subplots(figsize=(6.9, 4.4), facecolor="#ffffff")
    ax.axis("off")
    ax.set_aspect("equal")
    ax.set_xlim(min(xs) - x_pad - left_pad, max(xs) + x_pad)
    ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad + 0.4)

    for d in range(1, max_depth + 1):
        ax.text(min(xs) - x_pad - left_pad + 0.15, -d, f"hop {d}",
                ha="left", va="center", fontsize=8.5, color=INK_DIM,
                fontweight="bold", style="italic")
        ax.plot([min(xs) - x_pad - left_pad + 0.65, min(xs) - x_pad + 0.15],
                [-d, -d], color=LINE, lw=1.0, ls=":", zorder=0)

    for cid, u, v in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=TREE_ACCENT, lw=2.2,
                                     shrinkA=15, shrinkB=15, mutation_scale=14),
                    zorder=2)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.12, cid.replace("S0_", ""), ha="center",
                fontsize=7.4, color=INK_DIM, fontfamily="monospace",
                zorder=3, bbox=dict(fc="#ffffff", ec="none", pad=0.4))

    for node in pos:
        x, y = pos[node]
        is_root = node == 0
        ax.add_patch(Circle((x, y), 0.26,
                             facecolor=(TREE_ACCENT if is_root else "#ffffff"),
                             edgecolor=INK, lw=1.4, zorder=4))
        ax.text(x, y, str(node), ha="center", va="center", fontsize=10.5,
                fontweight="bold", zorder=5,
                color=("#ffffff" if is_root else INK))

    ax.text(0.5, 1.04,
            f"source 0's broadcast tree — {len(edges)} edges, "
            f"{len(pos)} nodes reached "
            f"(of {stats['policy_entries']} total entries, "
            f"{stats['trees_built']} trees)",
            transform=ax.transAxes, ha="center", fontsize=8.8, color=INK_DIM,
            style="italic")

    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="#ffffff")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default="ring_cluster")
    ap.add_argument("--size", type=float, default=None)
    args = ap.parse_args()

    device = torch.device("cpu")
    pyg = torch.load(DATA / "pyg_agent_test.pt", map_location="cpu",
                     weights_only=False)
    data = pick_instance(pyg, args.topology, args.size)
    ms = float(data.message_size_bytes)
    print(f"Instance: {args.topology} @ {ms:,.0f} B")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bu, bv = draw_topology(data, OUT_DIR / "worked_example_topology")
    print(f"Saved: {OUT_DIR / 'worked_example_topology'}.png / .svg")

    draw_tree(data, OUT_DIR / "worked_example_tree", device)
    print(f"Saved: {OUT_DIR / 'worked_example_tree'}.png / .svg")

    print(f"\nBusiest edge in this instance: {bu} -> {bv}")
    print("(must match the EDGE line in worked_example.py's [2] block — "
          "if not, --size resolved to a different graph in the two "
          "scripts; pin the exact same message_size_bytes in both.)")


if __name__ == "__main__":
    main()
