"""
load_priors.py — analytic, topology-general load priors for the Agent.

WHY (ties to the measured failure mode):
The Agent is blind on switch edges of UNSEEN topology families
(r=0.03 -> 33x downstream CCT gap on DGX2 holdout), and upweighting
proved it is not class imbalance: the missing ingredient is
INFORMATION about an unseen family's load regime. This module injects
that information analytically instead of hoping message passing
rediscovers it:

  A switch edge's load is governed by how many (source, destination)
  demand pairs are FORCED through it — i.e. demand-weighted edge
  betweenness. That is a computation, not a learned pattern, so it
  transfers to families the model has never seen: a model trained on
  NDv2/clos switches can carry the relation "huge betweenness -> huge
  load" to DGX2, because the relation itself is family-independent.

Two deliverables per graph:

1. FEATURES (2 new edge columns):
     log1p_sp_load_latency  — chunks crossing this edge if every
                              source's chunk followed latency-weighted
                              shortest paths to every participant
     log1p_sp_load_hops     — same under hop-count shortest paths
   Two variants because capacity/latency-aware and purely structural
   routing genuinely differ on switched fabrics; giving both lets the
   model interpolate.

2. RESIDUAL TARGET (SimNet's trick, ported):
     load_residual_target = log1p(true_load) - log1p(sp_load_latency)
   The model predicts a CORRECTION over the analytic baseline. Under
   OOD failure the prediction degrades to "analytic + noise" instead
   of "blind guess" — the baseline is computable at inference from
   topology alone, so absolute load is always recoverable:
     pred_absolute = pred_residual + log1p(sp_load_latency).
"""

import torch
import networkx as nx


def compute_sp_load_priors(edge_index, latency, num_nodes,
                           source_indices):
    """
    Demand-weighted shortest-path edge load for AllGather demand
    (every source -> every other source; relays are pass-through).

    Args:
        edge_index:     LongTensor [2, E] (directed)
        latency:        FloatTensor [E] — per-edge latency (relative
                        or absolute; only the ORDERING matters here)
        num_nodes:      int
        source_indices: iterable of node ids that own a chunk
                        (is_source == 1). Falls back to all nodes if
                        empty — with a stats flag, never silently.

    Returns:
        sp_load_latency: FloatTensor [E] — chunk count per edge under
                         latency-weighted Dijkstra routing
        sp_load_hops:    FloatTensor [E] — same under hop-count
        stats:           dict (unreachable pairs, fallback flags) —
                         surface these; unreachable demand pairs mean
                         the topology extraction is broken upstream.
    """
    ei = edge_index.cpu().numpy()
    lat = latency.cpu().numpy()
    E = ei.shape[1]

    sources = sorted(int(s) for s in source_indices)
    used_fallback = False
    if not sources:
        sources = list(range(num_nodes))
        used_fallback = True

    def build_graph(weight_mode):
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        for k in range(E):
            u, v = int(ei[0, k]), int(ei[1, k])
            w = 1.0 if weight_mode == "hops" else float(lat[k]) + 1e-9
            # parallel edges: keep the cheapest, remember its index
            if G.has_edge(u, v):
                if w < G[u][v]["weight"]:
                    G[u][v].update(weight=w, idx=k)
            else:
                G.add_edge(u, v, weight=w, idx=k)
        return G

    def sp_load(weight_mode):
        G = build_graph(weight_mode)
        load = torch.zeros(E)
        unreachable = 0
        for s in sources:
            try:
                _, paths = nx.single_source_dijkstra(
                    G, s, weight="weight")
            except nx.NetworkXError:
                unreachable += len(sources) - 1
                continue
            for t in sources:
                if t == s:
                    continue
                path = paths.get(t)
                if path is None:
                    unreachable += 1
                    continue
                for a, b in zip(path[:-1], path[1:]):
                    load[G[a][b]["idx"]] += 1.0
        return load, unreachable

    sp_lat, unreach_lat = sp_load("latency")
    sp_hop, unreach_hop = sp_load("hops")

    stats = {
        "n_sources": len(sources),
        "n_demand_pairs": len(sources) * (len(sources) - 1),
        "unreachable_pairs_latency": unreach_lat,
        "unreachable_pairs_hops": unreach_hop,
        "source_fallback_all_nodes": used_fallback,
    }
    return sp_lat, sp_hop, stats


def validate_priors(sp_load, true_load, name=""):
    """
    Optional one-shot diagnostic to run BEFORE retraining: how much of
    the truth does the analytic prior alone already explain? Prints
    Pearson r between log1p(prior) and log1p(true). If this is high on
    the DGX2 holdout (where the model scored r=0.03), the experiment
    is well-motivated; if the prior itself correlates poorly there,
    expect variant (c) — 'even the analytic prior is not enough'.
    """
    a = torch.log1p(sp_load.float())
    b = true_load.float()          # already log1p in canonical
    if a.std() < 1e-9 or b.std() < 1e-9:
        print(f"[prior-check {name}] degenerate variance, skipped")
        return float("nan")
    r = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
    print(f"[prior-check {name}] corr(log1p(sp_load), true) = {r:.3f}")
    return r
