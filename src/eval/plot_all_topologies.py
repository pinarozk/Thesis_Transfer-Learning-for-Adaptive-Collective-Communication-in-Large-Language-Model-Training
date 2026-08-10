"""
plot_all_topologies.py — draw every topology family from its REAL
edge_index (smallest message_size instance per family, same selection
rule as print_topology_adjacency.py: connectivity doesn't depend on
message size, only physics does).

Same visual language as the worked-example / gallery slides: white
fill + navy outline = source, solid orange = relay/switch.

Output: one PNG per family + one combined grid, 300 dpi,
src/eval/plots/topo_<name>.png and topo_gallery_full.png.
"""

import sys
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

DATA = ROOT / "data" / "processed"
OUT_DIR = ROOT / "src" / "eval" / "plots"

INK = "#0f2a44"
INK_DIM = "#4d5b68"
SOURCE_EDGE = "#0f2a44"
RELAY_FILL = "#c9781f"
LINE = "#c7cdcb"

FAMILIES = [
    "DGX2_2_chassis",
    "ring_cluster",
    "switched_dual_star_n12_r4",
    "switched_swclust_2x8_r4",
    "switched_two_tier_l4x4_r8",
]

SEED = 7


def pick_instance(pyg, topology):
    cand = [d for d in pyg if str(d.topology_name) == topology]
    if not cand:
        raise SystemExit(f"No graphs named {topology!r}.")
    return min(cand, key=lambda d: float(d.message_size_bytes))


def build_graph(data):
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
    return G, is_source


def draw_one(ax, G, is_source, title, node_size_scale=1.0, label_fs=None):
    n = G.number_of_nodes()
    max_id_digits = len(str(n - 1))
    k = 1.6 / max(np.sqrt(n), 1)
    pos = nx.spring_layout(G, seed=SEED, k=k, iterations=300)

    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color=LINE, lw=max(0.5, 1.6 - n * 0.01),
                 alpha=0.75, zorder=1)

    if n <= 24:
        r = (0.052 if max_id_digits <= 1 else 0.062) * node_size_scale
        label_fs = label_fs or (8.6 if max_id_digits <= 1 else 7.2)
    else:
        r = 0.032 * node_size_scale
        label_fs = label_fs or 6.0
    for node in G.nodes():
        x, y = pos[node]
        if is_source[node]:
            ax.add_patch(Circle((x, y), r, facecolor="#ffffff",
                                 edgecolor=SOURCE_EDGE, lw=1.3, zorder=3))
            tcolor = INK
        else:
            ax.add_patch(Circle((x, y), r, facecolor=RELAY_FILL,
                                 edgecolor=INK, lw=1.0, zorder=3))
            tcolor = "#ffffff"
        if n <= 24:
            ax.text(x, y, str(node), ha="center", va="center",
                    fontsize=label_fs, fontweight="bold", color=tcolor,
                    zorder=4)

    ax.set_aspect("equal")
    ax.axis("off")
    n_src = int(is_source.sum())
    ax.set_title(
        f"{title}\nN={n}  ·  {n_src} source / {n - n_src} relay  ·  "
        f"{G.number_of_edges()} links",
        fontsize=10, color=INK, fontweight="bold", pad=8,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pyg = torch.load(DATA / "pyg_agent_test.pt", map_location="cpu",
                     weights_only=False)

    graphs = {}
    for fam in FAMILIES:
        data = pick_instance(pyg, fam)
        G, is_source = build_graph(data)
        graphs[fam] = (G, is_source, float(data.message_size_bytes))
        print(f"{fam:32s} N={G.number_of_nodes():3d}  "
              f"E={G.number_of_edges():3d}  "
              f"msg_size={float(data.message_size_bytes):,.0f} B")

    # ---- individual figures ----
    for fam, (G, is_source, ms) in graphs.items():
        fig, ax = plt.subplots(figsize=(6.0, 5.2), facecolor="#ffffff")
        draw_one(ax, G, is_source, fam)
        legend = [
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#fff",
                        markeredgecolor=SOURCE_EDGE, markersize=9, label="source"),
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=RELAY_FILL,
                        markeredgecolor=INK, markersize=9, label="relay / switch"),
        ]
        ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.06),
                  ncol=2, frameon=False, fontsize=8.5)
        fig.tight_layout()
        out = OUT_DIR / f"topo_{fam}"
        for ext in ("png", "svg"):
            fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight",
                        facecolor="#ffffff")
        plt.close(fig)
        print(f"  saved {out}.png / .svg")

    # ---- combined gallery ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor="#ffffff")
    axes = axes.flatten()
    for ax, fam in zip(axes, FAMILIES):
        G, is_source, ms = graphs[fam]
        draw_one(ax, G, is_source, fam)
    axes[-1].axis("off")
    legend = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#fff",
                    markeredgecolor=SOURCE_EDGE, markersize=11, label="source (is_source=1)"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=RELAY_FILL,
                    markeredgecolor=INK, markersize=11, label="relay / switch"),
    ]
    fig.legend(handles=legend, loc="lower right", bbox_to_anchor=(0.98, 0.06),
               frameon=False, fontsize=11)
    fig.suptitle("All 5 test-set topology families — real edge_index, "
                  "smallest-message-size instance per family",
                  fontsize=13, color=INK, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUT_DIR / "topo_gallery_full"
    for ext in ("png", "svg"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="#ffffff")
    plt.close(fig)
    print(f"\nsaved {out}.png / .svg")


if __name__ == "__main__":
    main()
