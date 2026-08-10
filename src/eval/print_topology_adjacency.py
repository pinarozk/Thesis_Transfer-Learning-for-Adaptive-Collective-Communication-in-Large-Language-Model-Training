"""
print_topology_adjacency.py — exact ground-truth wiring for one
topology family, straight from the real data.

WHY THIS SCRIPT EXISTS: DGX2 and ring-cluster come from TE-CCL's own
topology class and your own build_merged_dataset.py respectively —
neither was ever fully transcribed into this conversation, so
reconstructing their wiring from memory would risk exactly the kind
of silent error this project has spent a month hunting down. This
script reads the real edge_index and is_source flags and prints
everything needed to draw the family correctly.

Usage:
    python print_topology_adjacency.py --topology DGX2_2_chassis
    python print_topology_adjacency.py --topology ring_cluster
    python print_topology_adjacency.py --list          # see all names

Output: node count, source/relay split, and the full edge list
grouped by source node (undirected pairs deduplicated) — paste
straight into a diagram tool or send back for a drawing spec.
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

DATA = ROOT / "data" / "processed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default=None)
    ap.add_argument("--size", type=float, default=None,
                    help="message_size_bytes; default = smallest "
                         "available (usually the simplest instance)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--source", default="test",
                    choices=["train", "val", "test", "dataset"])
    args = ap.parse_args()

    pyg = torch.load(DATA / f"pyg_agent_{args.source}.pt",
                     map_location="cpu", weights_only=False)

    if args.list:
        names = sorted({str(d.topology_name) for d in pyg})
        for n in names:
            n_nodes = next(int(d.x.size(0)) for d in pyg
                          if str(d.topology_name) == n)
            print(f"{n:32s} N={n_nodes}")
        return

    if not args.topology:
        raise SystemExit("Pass --topology NAME or --list to see names.")

    cand = [d for d in pyg if str(d.topology_name) == args.topology]
    if not cand:
        raise SystemExit(f"No graphs named '{args.topology}'. "
                         f"Run with --list.")
    # smallest message size = same connectivity, simplest to reason about
    # (connectivity does not depend on message size, only physics does)
    data = min(cand, key=lambda d: float(d.message_size_bytes))

    node_names = list(data.node_feature_names)
    src_i = node_names.index("is_source")
    is_source = (data.x[:, src_i].numpy() > 0.5)
    N = int(data.x.size(0))
    ei = data.edge_index.numpy()
    E = ei.shape[1]

    sources = [i for i in range(N) if is_source[i]]
    relays = [i for i in range(N) if not is_source[i]]

    print("=" * 60)
    print(f"{args.topology}  (message_size = "
         f"{float(data.message_size_bytes):,.0f} B)")
    print("=" * 60)
    print(f"nodes N = {N}    directed edges E = {E}")
    print(f"sources ({len(sources)}): {sources}")
    print(f"relays/switches ({len(relays)}): {relays}")

    # deduplicate to undirected pairs for a clean adjacency listing
    pairs = set()
    for k in range(E):
        u, v = int(ei[0, k]), int(ei[1, k])
        pairs.add(tuple(sorted((u, v))))

    print(f"\nundirected physical links: {len(pairs)}")
    print("(every physical link appears as TWO directed edges in the "
         "graph — src->dst and dst->src)")

    by_node = defaultdict(list)
    for u, v in pairs:
        by_node[u].append(v)
        by_node[v].append(u)

    print("\nadjacency (node: neighbours):")
    for n in range(N):
        role = "source" if is_source[n] else "RELAY"
        nbrs = sorted(by_node.get(n, []))
        print(f"  {n:3d} [{role:6s}]  degree={len(nbrs):3d}  -> {nbrs}")

    print("\nfull undirected edge list (paste-ready):")
    for u, v in sorted(pairs):
        tag = "" if (is_source[u] and is_source[v]) else "  <- touches a relay"
        print(f"  {u:3d} -- {v:3d}{tag}")


if __name__ == "__main__":
    main()
