"""
schedule_bridge.py — the missing link of the COCA loop.

The Agent lives in the canonical world (PyG graphs, edge load
predictions); SimNet and the real simulator live in the schedule world
(nx.DiGraph topologies + PolicyEntry lists). Steps 3 and 4 of the COCA
pipeline are impossible without an explicit, feasibility-guaranteed
DECODER between them. This module provides:

  decode_allgather(data, load_pred, sched_pred, threshold)
      Edge load predictions -> executable AllGather schedule:
      for every source node, a broadcast tree is grown over the
      SELECTED edge set using Dijkstra with weight (load_max -
      load_pred) -- i.e. the maximum-predicted-load tree (routing ->
      load regression, see agent_gnn.py fix #8: load_pred is a log1p
      edge-load regression output, NOT a [0,1] probability, so there
      is no "-log(p)" to take; higher predicted load = more preferred
      edge = lower Dijkstra cost via this affine flip, which keeps
      weights non-negative as Dijkstra requires).
      FEASIBILITY GUARANTEE: if the selected subgraph does not reach
      every node, the search transparently falls back to the full
      graph with heavily penalized unselected edges — the schedule is
      always complete, and the number of fallback edges is reported
      (a diagnostic of Agent quality). sched_pred orders sibling
      transfers (earlier predicted first-use -> earlier dependency
      position).

  to_simulator_topology(data)
      Canonical graph -> simulator nx.DiGraph. Physics (capacity,
      latency) is recovered from the log-absolute edge features; GPU
      node attributes are set to simulator defaults.

  featurize_for_simnet(topo, policy, chunk_size_bytes)
      (topology, schedule) -> SimNet input Data with lb_log10, reusing
      the EXACT feature builders of build_simnet_pyg.py (no second
      implementation to drift out of sync).

  run_ground_truth(topo, policy)
      Real-simulator makespan for a decoded schedule.

UNIT ASSUMPTION (read this once, loudly): canonical `capacity` is
converted to bits/sec via CAPACITY_TO_BPS. Set it to match your data
generation (e.g. 1.0 if capacities were already bps, 8e9 if GB/s).
The value used is printed at import time.
"""

import sys
import math
from pathlib import Path

import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "CCL_Simulator"))
sys.path.append(str(ROOT / "src" / "data"))

from simulator_wrapper import run_simulator          # noqa: E402
from simcore import PolicyEntry                      # noqa: E402
from src.data.build_simnet_pyg import (              # noqa: E402
    build_node_features,
    build_edge_tensors,
    build_global_features,
)

# ---- unit contract (adjust once, verify against your generator) ----
CAPACITY_TO_BPS = 8e9
print(f"[schedule_bridge] CAPACITY_TO_BPS = {CAPACITY_TO_BPS} "
      f"(verify against your topology generator units)")

EPS = 1e-9

# Must match train_agent.py / eval_agent.py's LOAD_USED_THRESHOLD:
# log1p(1) == the model predicting >=1 whole chunk crossed the edge.
LOAD_USED_THRESHOLD = math.log1p(1.0)


# ============================================================
# Physics recovery from canonical features
# ============================================================

def _feat_idx(data, name):
    names = data.edge_feature_names
    if name not in names:
        raise KeyError(f"'{name}' not in edge features {names}")
    return names.index(name)


def _feat_idx_node(data, name):
    names = data.node_feature_names
    if name not in names:
        raise KeyError(f"'{name}' not in node features {names}")
    return names.index(name)


def recover_physics(data):
    """capacity (raw units) and latency (sec) per directed edge."""
    log_cap = data.edge_attr[:, _feat_idx(data, "log10_capacity")]
    log_lat = data.edge_attr[:, _feat_idx(data, "log10_latency")]
    capacity = torch.pow(10.0, log_cap).numpy()
    latency = torch.pow(10.0, log_lat).numpy()
    # log10 sentinel 0.0 encodes 'zero/unknown' -> 10^0 = 1; map back
    capacity[log_cap.numpy() == 0.0] = 0.0
    latency[log_lat.numpy() == 0.0] = 0.0
    return capacity, latency


# ============================================================
# Simulator topology
# ============================================================

def to_simulator_topology(data):
    capacity, latency = recover_physics(data)
    ei = data.edge_index.numpy()
    n = int(data.x.size(0))

    G = nx.DiGraph()
    for i in range(n):
        G.add_node(
            f"GPU{i}",
            type="gpu",
            num_qps=2,
            quantum_packets=1,
            tx_proc_delay=0.0,
            gpu_store_delay=0.0,
        )
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        rate = float(capacity[k]) * CAPACITY_TO_BPS
        if rate <= 0:
            rate = 1e9  # degenerate edge guard; should not happen
        G.add_edge(f"GPU{u}", f"GPU{v}",
                   link_rate_bps=rate,
                   prop_delay=float(latency[k]))
    return G


# ============================================================
# Decoder: probabilities -> executable AllGather schedule
# ============================================================

def agent_per_source_chunk_bytes(data):
    """
    The AllGather PolicyEntry chunk_size_bytes decode_allgather actually
    builds -- the single source of truth for "how many bytes cross one
    edge", shared with teacher_replay._per_source_chunk_bytes (same
    domain-conditional derivation; see that docstring for the full
    numeric proof on DGX2_2_chassis / ring_cluster).

    Callers that need a chunk size OUTSIDE decode_allgather (e.g.
    featurize_for_simnet's global log2_chunk_size feature in
    calibrate_simnet.py / finetune_agent.py) MUST use this, not raw
    data.message_size_bytes: for source_domain=="teccl" samples the raw
    field is n_sources times too large. Before this fix, Step 3/4 built
    SimNet's global chunk-size feature from the raw field while the
    policy's actual edge-level bytes (and the real simulator makespan)
    used the correct, n_sources-divided value -- a silent, systematic
    ~32x-off global feature on exactly the DGX2/TE-CCL domain, never
    present in SimNet's own training data (generate_simnet_dataset.py
    always uses one consistent chunk_size for both). This fed Step 4's
    RL reward (surrogate_reward) an out-of-distribution global input on
    every TE-CCL-origin graph.
    """
    is_source_i = _feat_idx_node(data, "is_source")
    is_source = data.x[:, is_source_i].numpy() > 0.5
    n_sources = max(int(is_source.sum()), 1)
    if str(getattr(data, "source_domain", "")) == "teccl":
        return float(data.message_size_bytes) / n_sources
    return float(data.message_size_bytes)


def decode_allgather(data, load_pred, sched_pred, threshold=None):
    """
    Returns (policy_entries, stats).
    stats: selected_edges, fallback_edges, trees_built.
    threshold: cut in log1p(load) space (default LOAD_USED_THRESHOLD
    == log1p(1), matching train_agent.py/eval_agent.py).
    """
    if threshold is None:
        threshold = LOAD_USED_THRESHOLD
    load_pred = np.asarray(load_pred, dtype=float)
    sched_pred = np.asarray(sched_pred, dtype=float)
    ei = data.edge_index.numpy()
    n_nodes = int(data.x.size(0))
    n_edges = ei.shape[1]
    switches = set(data.switch_indices or [])

    is_source_i = _feat_idx_node(data, "is_source")
    is_source = data.x[:, is_source_i].numpy() > 0.5
    sources = [i for i in range(n_nodes) if is_source[i]]
    chunk = agent_per_source_chunk_bytes(data)

    selected = load_pred >= threshold

    # weighted digraph: cost = (load_max - load_pred), i.e. HIGHER
    # predicted load -> LOWER cost -> more preferred by Dijkstra (the
    # maximum-predicted-load tree). Unselected edges get a large
    # additive penalty -> used ONLY when the selected subgraph cannot
    # reach a node (repair, not preference).
    PENALTY = 1e3
    load_max = float(load_pred.max()) if n_edges else 0.0
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    for k in range(n_edges):
        u, v = int(ei[0, k]), int(ei[1, k])
        w = (load_max - float(load_pred[k])) + EPS
        if not selected[k]:
            w += PENALTY
        # tiny tie-break: prefer edges the Agent schedules earlier
        w += 1e-3 * float(sched_pred[k])
        if G.has_edge(u, v):
            if w < G[u][v]["weight"]:
                G[u][v].update(weight=w, idx=k)
        else:
            G.add_edge(u, v, weight=w, idx=k)

    policy = []
    fallback_edges = set()
    trees_built = 0
    cid = 0

    # is_source (node_feat col 0) is the ground-truth-derived
    # functional-role signal (see build_canonical_dataset.get_source_
    # indices): which nodes actually originate a chunk. Prefer it over
    # switch_indices directly -- switch_indices marks the structural/
    # expected complement, is_source is what the schedule actually
    # used, and the two can diverge (that mismatch is exactly what
    # build_sample's own coverage check guards against upstream).
    # (sources/n_sources already computed above, for the chunk-size fix.)

    for s in sources:
        # max-predicted-load shortest-path tree from s
        try:
            _, paths = nx.single_source_dijkstra(G, s,
                                                 weight="weight")
        except Exception as e:
            raise RuntimeError(
                f"Decode failed from source {s}: {e}"
            )
        # union of s->t paths is a tree (Dijkstra SP tree)
        parent_entry = {s: None}
        # visit nodes in nondecreasing path length so parents exist
        order = sorted(paths.keys(),
                       key=lambda t: len(paths[t]))
        for t in order:
            if t == s:
                continue
            path = paths[t]
            u, v = path[-2], path[-1]
            k = G[u][v]["idx"]
            if not selected[k]:
                fallback_edges.add(k)
            if v in parent_entry:      # already reached via tree
                continue
            entry_id = f"S{s}_C{cid}"
            cid += 1
            dep = parent_entry.get(u)
            policy.append(PolicyEntry(
                entry_id,
                f"GPU{u}", f"GPU{v}",
                qpid=0,
                rate="Max",
                chunk_size_bytes=chunk,
                path=[f"GPU{u}", f"GPU{v}"],
                time=0.0,
                dependency=[dep] if dep else [],
            ))
            parent_entry[v] = entry_id
        trees_built += 1

    stats = {
        "selected_edges": int(selected.sum()),
        "total_edges": n_edges,
        "fallback_edges": len(fallback_edges),
        "trees_built": trees_built,
        "policy_entries": len(policy),
    }
    return policy, stats


# ============================================================
# SimNet featurization (reuses pipeline builders — zero drift)
# ============================================================

def featurize_for_simnet(topo, policy, chunk_size_bytes):
    nodes = list(topo.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    x = build_node_features(topo, nodes)
    edge_index, edge_attr, extras = build_edge_tensors(
        topo, node_to_idx, policy
    )
    sample_like = {
        "chunk_size_bytes": chunk_size_bytes,
        "policy": policy,
        "num_policy_entries": len(policy),
    }
    u = build_global_features(sample_like, topo, extras)

    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u)
    lb = max(float(extras["analytic_lower_bound"]), 1e-12)
    g.analytic_lower_bound = lb
    g.lb_log10 = torch.tensor([math.log10(lb)], dtype=torch.float)
    return g


# ============================================================
# Ground truth
# ============================================================

MAX_PACKETS_PER_ENTRY = 4000
MIN_PACKET_SIZE_BYTES = 1500


def _safe_packet_size(policy):
    """Packet size adaptive to message size, capping total simpy
    packet-events per policy entry. tx_proc_delay/gpu_store_delay are
    0.0 for all decoded topologies (see to_simulator_topology), and
    port.py's _service_time is size_bytes/rate_bps with no fixed
    per-packet overhead -- so total transfer time only depends on
    total bytes/rate, not packetization. Real message sizes up to 1e9
    bytes at the default 1500B MTU produce ~7e5 packets/edge, which
    across dozens of edges and multiple decode calls exhausts memory
    (this cap was added after exactly that OOM). Increasing packet
    size for large messages is therefore free: it does not change the
    makespan, only the event count.
    """
    if not policy:
        return MIN_PACKET_SIZE_BYTES
    largest_chunk = max(float(e.chunk_size_bytes) for e in policy)
    return max(MIN_PACKET_SIZE_BYTES,
               math.ceil(largest_chunk / MAX_PACKETS_PER_ENTRY))


def run_ground_truth(topo, policy):
    """Real-simulator makespan (seconds)."""
    packet_size_bytes = _safe_packet_size(policy)
    makespan, _ = run_simulator(topo, policy,
                                 packet_size_bytes=packet_size_bytes)
    if not (makespan > 0 and math.isfinite(makespan)):
        raise RuntimeError(f"Invalid simulator makespan: {makespan}")
    return float(makespan)


# ============================================================
# One-call convenience: Agent output -> everything downstream
# ============================================================

@torch.no_grad()
def agent_to_schedule(agent, data, device, threshold=None,
                      load_target_mode="absolute"):
    """Frozen-Agent inference + decode. Returns
    (topo, policy, load_pred, sched_pred, decode_stats).
    load_target_mode: "residual" if the checkpoint was trained against
    y_load_residual (see load_priors.py / train_agent.py LOAD_TARGET)
    -- the analytic sp_load_log1p baseline is added back so load_pred
    lands in the same absolute log1p(load) space decode_allgather and
    `threshold` expect. Read this from the checkpoint's saved config
    (eval_agent.load_model's third return value), don't guess."""
    if threshold is None:
        threshold = LOAD_USED_THRESHOLD
    u = data.u
    if u is not None and u.dim() == 1:
        u = u.unsqueeze(0)
    load_pred, _, sched = agent.predict(
        data.x.to(device), data.edge_index.to(device),
        data.edge_attr.to(device), batch=None, u=u.to(device),
        load_threshold=threshold,
    )
    if load_target_mode == "residual":
        load_pred = load_pred + data.sp_load_log1p.to(device)
    load_pred = load_pred.cpu().numpy()
    sched = sched.cpu().numpy()
    topo = to_simulator_topology(data)
    policy, stats = decode_allgather(data, load_pred, sched, threshold)
    return topo, policy, load_pred, sched, stats
