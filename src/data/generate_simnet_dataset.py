"""
SimNet training-data generator (thesis-ready).

Fixes vs. previous version
--------------------------
1. REPRODUCIBILITY: the old script used the global `random` module with
   NO seed — the dataset could never be regenerated. Every sample now
   has a deterministic hashlib-derived seed and its own local RNG; the
   seed is stored in the sample.
2. DISTRIBUTION MATCH (critical for SimNet's actual job): SimNet must
   score Agent/ILP schedules, but the old policies were exclusively
   SHORTEST-PATH random transfers. ILP schedules use non-shortest
   detours, broadcast trees, and heavy parallelism — a surrogate that
   never saw such inputs is unreliable exactly where it matters. Added:
     - 'detour'         : transfers routed over k-th shortest paths
     - 'allgather_tree' : BFS broadcast trees from every source
                          (structurally closest to teacher schedules)
   plus the original random/fanout/fanin/chain for coverage of BAD
   schedules (the surrogate must know what slow looks like, not only
   what fast looks like — quality diversity).
3. SIZE RANGE EXTENDED to {4, 6, 8, 12, 16} so the converter can build
   a size-extrapolation split (train small, test large) — the SimNet
   counterpart of the Agent's transfer claim.
4. INCREMENTAL SAVE + failed-case log (same pattern as the teacher
   generator): long simulator runs survive interruptions.
5. FULL METADATA per sample: seed, topology params, policy params —
   everything needed for group-based splitting downstream.
"""

import hashlib
import random
import sys
import time
from pathlib import Path

import torch
import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "CCL_Simulator"))

from simulator_wrapper import run_simulator
from simcore import PolicyEntry


# ============================================================
# Config
# ============================================================

NUM_SAMPLES = 3000
SAVE_PATH = Path("data/raw/simnet_samples.pt")
FAILED_PATH = Path("data/raw/simnet_failed.pt")

GLOBAL_TAG = "simnet_v2"          # bump to regenerate from scratch

NODE_COUNTS = [4, 6, 8, 12, 16]
LINK_RATES = [25e9, 50e9, 100e9, 200e9]
CHUNK_MB_OPTIONS = [1, 4, 16, 64, 128]

TOPOLOGY_TYPES = ["line", "ring", "star", "bottleneck", "random"]
POLICY_TYPES = [
    "random", "fanout", "fanin", "chain",
    "detour",          # fix #2: non-shortest paths
    "allgather_tree",  # fix #2: teacher-schedule-like structure
]


def deterministic_seed(*parts) -> int:
    key = "|".join(str(p) for p in parts)
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


# ============================================================
# Topologies (all take a local rng — never the global module)
# ============================================================

def add_gpu_nodes(G, n, rng):
    for i in range(n):
        G.add_node(
            f"GPU{i}",
            type="gpu",
            num_qps=rng.choice([1, 2, 4]),
            quantum_packets=rng.choice([1, 2, 4]),
            tx_proc_delay=0.0,
            gpu_store_delay=0.0,
        )


def add_edge_pair(G, u, v, rng, rate=None, delay=None):
    rate = rate if rate is not None else rng.choice(LINK_RATES)
    delay = delay if delay is not None else rng.uniform(0.0, 0.002)
    G.add_edge(u, v, link_rate_bps=rate, prop_delay=delay)
    G.add_edge(v, u, link_rate_bps=rate, prop_delay=delay)


def build_line(n, rng):
    G = nx.DiGraph()
    add_gpu_nodes(G, n, rng)
    for i in range(n - 1):
        add_edge_pair(G, f"GPU{i}", f"GPU{i+1}", rng)
    G.graph["topology_type"] = "line"
    return G


def build_ring(n, rng):
    G = build_line(n, rng)
    add_edge_pair(G, f"GPU{n-1}", "GPU0", rng)
    G.graph["topology_type"] = "ring"
    return G


def build_star(n, rng):
    G = nx.DiGraph()
    add_gpu_nodes(G, n, rng)
    for i in range(1, n):
        add_edge_pair(G, "GPU0", f"GPU{i}", rng)
    G.graph["topology_type"] = "star"
    return G


def build_bottleneck(n, rng):
    G = nx.DiGraph()
    add_gpu_nodes(G, n, rng)
    split = n // 2
    for i in range(split - 1):
        add_edge_pair(G, f"GPU{i}", f"GPU{i+1}", rng, rate=100e9)
    for i in range(split, n - 1):
        add_edge_pair(G, f"GPU{i}", f"GPU{i+1}", rng, rate=100e9)
    add_edge_pair(G, f"GPU{split-1}", f"GPU{split}", rng,
                  rate=25e9, delay=rng.uniform(0.001, 0.004))
    G.graph["topology_type"] = "bottleneck"
    return G


def build_random_topo(n, rng):
    G = nx.DiGraph()
    add_gpu_nodes(G, n, rng)
    for i in range(n - 1):
        add_edge_pair(G, f"GPU{i}", f"GPU{i+1}", rng)
    possible = [
        (f"GPU{i}", f"GPU{j}")
        for i in range(n) for j in range(i + 1, n)
        if abs(i - j) > 1
    ]
    rng.shuffle(possible)
    for u, v in possible[: rng.randint(1, min(len(possible), n))]:
        add_edge_pair(G, u, v, rng)
    G.graph["topology_type"] = "random"
    return G


TOPO_BUILDERS = {
    "line": build_line,
    "ring": build_ring,
    "star": build_star,
    "bottleneck": build_bottleneck,
    "random": build_random_topo,
}


# ============================================================
# Policies
# ============================================================

def sp_entry(chunk_id, topo, src, dst, chunk_size, t, rng,
             dependency=None, path=None):
    if path is None:
        path = nx.shortest_path(topo, source=src, target=dst)
    return PolicyEntry(
        chunk_id, src, dst,
        qpid=rng.randint(0, 1),
        rate="Max",
        chunk_size_bytes=chunk_size,
        path=path,
        time=t,
        dependency=dependency or [],
    )


def gen_random(topo, chunk_size, rng):
    nodes = list(topo.nodes())
    policy = []
    for i in range(rng.randint(2, min(10, len(nodes) * 2))):
        src = rng.choice(nodes)
        dst = rng.choice([n for n in nodes if n != src])
        policy.append(sp_entry(f"R{i}", topo, src, dst, chunk_size,
                               rng.uniform(0.0, 1.0), rng))
    return policy


def gen_fanout(topo, chunk_size, rng):
    nodes = list(topo.nodes())
    src = rng.choice(nodes)
    dsts = rng.sample([n for n in nodes if n != src],
                      k=rng.randint(2, min(4, len(nodes) - 1)))
    return [sp_entry(f"FO{i}", topo, src, d, chunk_size, 0.0, rng)
            for i, d in enumerate(dsts)]


def gen_fanin(topo, chunk_size, rng):
    nodes = list(topo.nodes())
    dst = rng.choice(nodes)
    srcs = rng.sample([n for n in nodes if n != dst],
                      k=rng.randint(2, min(4, len(nodes) - 1)))
    return [sp_entry(f"FI{i}", topo, s, dst, chunk_size, 0.0, rng)
            for i, s in enumerate(srcs)]


def gen_chain(topo, chunk_size, rng):
    nodes = list(topo.nodes())
    chain = rng.sample(nodes, rng.randint(3, min(5, len(nodes))))
    policy, prev = [], None
    for i in range(len(chain) - 1):
        cid = f"CH{i}"
        policy.append(sp_entry(cid, topo, chain[i], chain[i + 1],
                               chunk_size, 0.0, rng,
                               dependency=[prev] if prev else []))
        prev = cid
    return policy


def gen_detour(topo, chunk_size, rng, k_max=4):
    """
    Fix #2a: transfers on deliberately NON-shortest paths. ILP schedules
    frequently route around congested links; a surrogate trained only
    on shortest paths extrapolates blindly on such inputs.
    """
    nodes = list(topo.nodes())
    policy = []
    for i in range(rng.randint(2, 8)):
        src = rng.choice(nodes)
        dst = rng.choice([n for n in nodes if n != src])
        try:
            gen = nx.shortest_simple_paths(topo, src, dst)
            paths = []
            for _, p in zip(range(k_max), gen):
                paths.append(p)
        except nx.NetworkXNoPath:
            continue
        path = paths[rng.randrange(len(paths))]  # any of top-k paths
        policy.append(sp_entry(f"DT{i}", topo, src, dst, chunk_size,
                               rng.uniform(0.0, 0.5), rng, path=path))
    if not policy:
        raise ValueError("detour policy empty")
    return policy


def gen_allgather_tree(topo, chunk_size, rng):
    """
    Fix #2b: BFS broadcast tree from EVERY source — structurally the
    closest cheap approximation of an AllGather teacher schedule
    (every node's chunk reaches every other node along tree edges).
    Chunks from the same source share a dependency chain along each
    root-to-leaf branch.
    """
    policy = []
    cid = 0
    for src in topo.nodes():
        tree = nx.bfs_tree(topo, src)
        # send src's chunk along each tree edge; parent transfer is a
        # dependency of the child transfer
        dep_of_node = {src: None}
        for u, v in nx.bfs_edges(topo, src):
            this_id = f"AG{cid}"
            cid += 1
            dep = dep_of_node.get(u)
            policy.append(sp_entry(
                this_id, topo, u, v, chunk_size, 0.0, rng,
                dependency=[dep] if dep else [],
                path=[u, v],
            ))
            dep_of_node[v] = this_id
        _ = tree
    return policy


POLICY_GENERATORS = {
    "random": gen_random,
    "fanout": gen_fanout,
    "fanin": gen_fanin,
    "chain": gen_chain,
    "detour": gen_detour,
    "allgather_tree": gen_allgather_tree,
}


# ============================================================
# Main
# ============================================================

def main():
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if SAVE_PATH.exists():
        dataset = torch.load(SAVE_PATH, weights_only=False)
        print(f"Loaded existing dataset: {len(dataset)} samples")
    else:
        dataset = []

    failed = (torch.load(FAILED_PATH, weights_only=False)
              if FAILED_PATH.exists() else [])

    done_ids = {s["sample_id"] for s in dataset}

    print("=" * 70)
    print("SimNet dataset generation (deterministic, incremental)")
    print("Target samples:", NUM_SAMPLES)
    print("Node counts:", NODE_COUNTS)
    print("Policy types:", POLICY_TYPES)
    print("=" * 70)

    for sample_id in range(NUM_SAMPLES):
        if sample_id in done_ids:
            continue

        seed = deterministic_seed(GLOBAL_TAG, sample_id)
        rng = random.Random(seed)

        n = rng.choice(NODE_COUNTS)
        topo_type = rng.choice(TOPOLOGY_TYPES)
        policy_type = rng.choice(POLICY_TYPES)
        chunk_mb = rng.choice(CHUNK_MB_OPTIONS)
        chunk_size = chunk_mb * 1024 * 1024

        try:
            topo = TOPO_BUILDERS[topo_type](n, rng)
            policy = POLICY_GENERATORS[policy_type](
                topo, chunk_size, rng
            )

            t0 = time.time()
            makespan, tx_times = run_simulator(topo, policy)
            sim_time = time.time() - t0

            dataset.append({
                "sample_id": sample_id,
                "seed": seed,
                "topology": topo,
                "policy": policy,
                "completion_time": float(makespan),
                "tx_times": tx_times,
                "topology_type": topo_type,
                "policy_type": policy_type,
                "num_nodes": topo.number_of_nodes(),
                "num_edges": topo.number_of_edges(),
                "chunk_mb": chunk_mb,
                "chunk_size_bytes": chunk_size,
                "num_policy_entries": len(policy),
                "sim_wall_time_sec": float(sim_time),
            })

            if len(dataset) % 50 == 0:
                torch.save(dataset, SAVE_PATH)
                torch.save(failed, FAILED_PATH)
                print(f"progress: {len(dataset)} saved "
                      f"(last: {topo_type}/{policy_type}, "
                      f"n={n}, {chunk_mb}MB, "
                      f"makespan={makespan:.4g})")

        except Exception as e:
            failed.append({
                "sample_id": sample_id, "seed": seed,
                "topology_type": topo_type,
                "policy_type": policy_type,
                "num_nodes": n, "chunk_mb": chunk_mb,
                "error": str(e),
            })

    torch.save(dataset, SAVE_PATH)
    torch.save(failed, FAILED_PATH)

    print("=" * 70)
    print("DONE — samples:", len(dataset), "| failed:", len(failed))

    from collections import Counter
    print("topology types:",
          Counter(s["topology_type"] for s in dataset))
    print("policy types:",
          Counter(s["policy_type"] for s in dataset))
    print("node counts:",
          Counter(s["num_nodes"] for s in dataset))


if __name__ == "__main__":
    main()
