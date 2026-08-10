"""
SimNet raw samples -> PyG dataset with group-based splits.

Fixes vs. previous version
--------------------------
1. GROUP-BASED SPLITS (the old converter produced NO split at all;
   a random split downstream would leak near-duplicates). Two modes:
     - "size_extrapolation" (default): train on n in {4,6,8},
       val n=12, test n=16. This is the SimNet counterpart of the
       Agent's "transfer across network scales" claim — a surrogate
       that only interpolates sizes it has seen cannot calibrate the
       Agent on larger topologies.
     - "type_holdout": whole topology types held out.
   Leakage assertions included.
2. PHYSICS-INFORMED FEATURES: the strongest predictor of makespan is
   per-edge serialization time (bytes / rate) — the old features made
   the network reconstruct it from two separately-logged quantities.
   Now provided directly, along with an ANALYTIC LOWER BOUND of the
   makespan as a global feature: max over edges of
   (first trigger + busy time + prop delay). The model then only has
   to learn the *contention/dependency correction* on top of a sound
   baseline — a much easier and better-generalizing target.
3. CONSISTENT LOG BASE: target was ln(makespan) while features used
   log10 — harmless numerically but confusing analytically. Everything
   is log10 now; raw makespan is stored for inverse transform.
4. TEMPORAL POLICY FEATURES: first/last trigger time per edge kept
   separately (the old avg collapsed schedule structure).
5. STATS FILE: train-split target mean/std saved for standardization
   at training time (computed on TRAIN only — no test statistics ever
   leak into normalization).
6. Feature-name metadata everywhere; fail-loud consistency checks.
"""

import math
import sys
from pathlib import Path
from collections import Counter, defaultdict

import torch
from torch_geometric.data import Data

ROOT_DIR = Path(__file__).resolve().parents[2]
SIMULATOR_DIR = ROOT_DIR / "CCL_Simulator"
sys.path.append(str(SIMULATOR_DIR))

RAW_PATH = Path("data/raw/simnet_samples.pt")

OUT_DIR = Path("data/processed")
OUT_FULL = OUT_DIR / "simnet_pyg.pt"
OUT_TRAIN = OUT_DIR / "simnet_pyg_train.pt"
OUT_VAL = OUT_DIR / "simnet_pyg_val.pt"
OUT_TEST = OUT_DIR / "simnet_pyg_test.pt"
OUT_STATS = OUT_DIR / "simnet_pyg_stats.pt"


# ------------------------------------------------------------
# Split configuration
# ------------------------------------------------------------
SPLIT_MODE = "size_extrapolation"   # or "type_holdout"

TRAIN_SIZES = {4, 6, 8}
VAL_SIZES = {12}
TEST_SIZES = {16}

TEST_TOPO_TYPES = {"bottleneck"}
VAL_TOPO_TYPES = {"star"}


NODE_FEATURE_NAMES = [
    "is_gpu", "num_qps", "quantum_packets",
    "tx_proc_delay", "gpu_store_delay",
]

EDGE_FEATURE_NAMES = [
    "log10_link_rate",
    "prop_delay",
    "num_transfers",
    "log10_total_bytes",
    "log10_serialization_time",   # fix #2: bytes / rate, the physics
    "first_trigger_time",         # fix #4
    "last_trigger_time",
    "avg_dependency_count",
]

GLOBAL_FEATURE_NAMES = [
    "log2_chunk_size",
    "log10_num_nodes",
    "num_edges",
    "density",
    "avg_log10_link_rate",
    "avg_prop_delay",
    "log10_total_policy_bytes",
    "num_policy_entries",
    "max_path_len",
    "log10_analytic_lower_bound",  # fix #2: sound baseline estimate
]

TARGET_NAME = "log10 of simulator makespan (raw stored separately)"


def _log10(x):
    return math.log10(max(x, 1e-12))


# ============================================================
# Feature builders
# ============================================================

def build_node_features(topo, nodes):
    x = []
    for node in nodes:
        d = topo.nodes[node]
        x.append([
            1.0 if d.get("type", "gpu") == "gpu" else 0.0,
            float(d.get("num_qps", 1)),
            float(d.get("quantum_packets", 1)),
            float(d.get("tx_proc_delay", 0.0)),
            float(d.get("gpu_store_delay", 0.0)),
        ])
    return torch.tensor(x, dtype=torch.float)


def aggregate_policy(topo, policy):
    stats = {
        (u, v): {
            "num_transfers": 0.0, "total_bytes": 0.0,
            "first_time": float("inf"), "last_time": 0.0,
            "dep_count": 0.0,
        }
        for u, v in topo.edges()
    }
    max_path_len = 0
    total_bytes = 0.0

    for entry in policy:
        path = entry.path
        max_path_len = max(max_path_len, len(path) - 1)
        size = float(entry.chunk_size_bytes)
        total_bytes += size
        t = float(entry.time)
        dep = getattr(entry, "dependency", []) or []

        for u, v in zip(path[:-1], path[1:]):
            if (u, v) not in stats:
                raise ValueError(
                    f"Policy uses edge not in topology: {(u, v)}"
                )
            s = stats[(u, v)]
            s["num_transfers"] += 1.0
            s["total_bytes"] += size
            s["first_time"] = min(s["first_time"], t)
            s["last_time"] = max(s["last_time"], t)
            s["dep_count"] += len(dep)

    return stats, max_path_len, total_bytes


def build_edge_tensors(topo, node_to_idx, policy):
    stats, max_path_len, total_policy_bytes = \
        aggregate_policy(topo, policy)

    edge_index, edge_attr = [], []
    lower_bound_terms = []
    link_rates, prop_delays = [], []

    for src, dst, d in topo.edges(data=True):
        s = stats[(src, dst)]
        rate = float(d.get("link_rate_bps", 0.0))
        delay = float(d.get("prop_delay", 0.0))
        link_rates.append(rate)
        prop_delays.append(delay)

        n_tx = s["num_transfers"]
        first_t = s["first_time"] if n_tx > 0 else 0.0
        last_t = s["last_time"]
        avg_dep = s["dep_count"] / n_tx if n_tx > 0 else 0.0

        # physics: time to serialize this edge's total traffic
        busy = (s["total_bytes"] * 8.0 / rate) if rate > 0 else 0.0

        # analytic lower bound contribution of this edge
        if n_tx > 0:
            lower_bound_terms.append(first_t + busy + delay)

        edge_index.append([node_to_idx[src], node_to_idx[dst]])
        edge_attr.append([
            _log10(rate + 1.0),
            delay,
            n_tx,
            _log10(s["total_bytes"] + 1.0),
            _log10(busy),
            first_t,
            last_t,
            avg_dep,
        ])

    lower_bound = max(lower_bound_terms) if lower_bound_terms else 1e-12

    edge_index = torch.tensor(edge_index,
                              dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    extras = {
        "link_rates": link_rates,
        "prop_delays": prop_delays,
        "max_path_len": max_path_len,
        "total_policy_bytes": total_policy_bytes,
        "analytic_lower_bound": lower_bound,
    }
    return edge_index, edge_attr, extras


def build_global_features(sample, topo, extras):
    n = float(topo.number_of_nodes())
    e = float(topo.number_of_edges())
    density = e / max(n * (n - 1), 1.0)
    rates = extras["link_rates"]
    delays = extras["prop_delays"]

    u = torch.tensor([[
        math.log2(float(sample["chunk_size_bytes"]) + 1.0),
        _log10(n),
        e,
        density,
        sum(_log10(r + 1.0) for r in rates) / max(len(rates), 1),
        sum(delays) / max(len(delays), 1),
        _log10(extras["total_policy_bytes"] + 1.0),
        float(sample.get("num_policy_entries", len(sample["policy"]))),
        float(extras["max_path_len"]),
        _log10(extras["analytic_lower_bound"]),
    ]], dtype=torch.float)
    return u


# ============================================================
# Sample conversion
# ============================================================

def make_pyg_graph(sample):
    topo = sample["topology"]
    policy = sample["policy"]
    makespan = float(sample["completion_time"])
    if not (makespan > 0 and math.isfinite(makespan)):
        raise ValueError(f"Invalid makespan: {makespan}")

    nodes = list(topo.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    x = build_node_features(topo, nodes)
    edge_index, edge_attr, extras = build_edge_tensors(
        topo, node_to_idx, policy
    )
    u = build_global_features(sample, topo, extras)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        u=u,
        y=torch.tensor([_log10(makespan)], dtype=torch.float),
    )

    data.raw_completion_time = makespan
    data.analytic_lower_bound = float(extras["analytic_lower_bound"])
    data.sample_id = int(sample["sample_id"])
    data.seed = int(sample["seed"])
    data.topology_type = str(sample.get("topology_type", "unknown"))
    data.policy_type = str(sample.get("policy_type", "unknown"))
    data.chunk_mb = float(sample.get("chunk_mb", -1))
    data.num_nodes_meta = int(topo.number_of_nodes())
    data.num_policy_entries = int(len(policy))

    data.node_feature_names = NODE_FEATURE_NAMES
    data.edge_feature_names = EDGE_FEATURE_NAMES
    data.global_feature_names = GLOBAL_FEATURE_NAMES
    data.target_name = TARGET_NAME

    # group key: the atomic unit for split integrity checks
    data.group_key = (f"{data.topology_type}"
                      f"::n{data.num_nodes_meta}")
    return data


# ============================================================
# Splits (fix #1)
# ============================================================

def split_dataset(graphs):
    train, val, test = [], [], []

    if SPLIT_MODE == "size_extrapolation":
        for g in graphs:
            n = g.num_nodes_meta
            if n in TEST_SIZES:
                test.append(g)
            elif n in VAL_SIZES:
                val.append(g)
            elif n in TRAIN_SIZES:
                train.append(g)
            # sizes outside all sets are dropped deliberately
        print(f"[split: size_extrapolation] "
              f"train={sorted(TRAIN_SIZES)}, "
              f"val={sorted(VAL_SIZES)}, "
              f"test={sorted(TEST_SIZES)}")

    elif SPLIT_MODE == "type_holdout":
        for g in graphs:
            t = g.topology_type
            if t in TEST_TOPO_TYPES:
                test.append(g)
            elif t in VAL_TOPO_TYPES:
                val.append(g)
            else:
                train.append(g)
        print(f"[split: type_holdout] "
              f"val={sorted(VAL_TOPO_TYPES)}, "
              f"test={sorted(TEST_TOPO_TYPES)}")
    else:
        raise ValueError(f"Unknown SPLIT_MODE: {SPLIT_MODE}")

    # leakage guarantee on the splitting attribute
    def keys(split, attr):
        return {getattr(g, attr) for g in split}

    attr = ("num_nodes_meta" if SPLIT_MODE == "size_extrapolation"
            else "topology_type")
    assert not keys(train, attr) & keys(test, attr), "train/test leak"
    assert not keys(train, attr) & keys(val, attr), "train/val leak"
    assert not keys(val, attr) & keys(test, attr), "val/test leak"
    print("  leakage check: PASSED")

    if not train or not val or not test:
        raise RuntimeError(
            f"Empty split (train={len(train)}, val={len(val)}, "
            f"test={len(test)}) — regenerate data with the required "
            f"sizes/types."
        )
    return train, val, test


# ============================================================
# Stats (fix #5)
# ============================================================

def compute_stats(train):
    y = torch.cat([g.y for g in train])
    lb = torch.tensor([g.analytic_lower_bound for g in train])
    lb_log = torch.log10(lb.clamp(min=1e-12))
    residual = y - lb_log   # what the model actually must learn

    return {
        "split_mode": SPLIT_MODE,
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "n_train": len(train),
        "node_feature_names": NODE_FEATURE_NAMES,
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "global_feature_names": GLOBAL_FEATURE_NAMES,
        "note": ("Standardize y with y_mean/y_std at training time. "
                 "Alternatively train on the residual over the "
                 "analytic lower bound (residual_mean/std provided) — "
                 "recommended for size extrapolation."),
    }


# ============================================================
# Summary
# ============================================================

def summarize(name, graphs):
    print(f"\n{name} — {len(graphs)} graphs")
    print("  topology types:",
          dict(Counter(g.topology_type for g in graphs)))
    print("  policy types:",
          dict(Counter(g.policy_type for g in graphs)))
    print("  node counts:",
          dict(Counter(g.num_nodes_meta for g in graphs)))
    y = torch.cat([g.y for g in graphs])
    print(f"  target log10(makespan): "
          f"mean={y.mean():.3f}, std={y.std():.3f}, "
          f"min={y.min():.3f}, max={y.max():.3f}")


# ============================================================
# Main
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = torch.load(RAW_PATH, weights_only=False)
    print("Raw samples:", len(raw))

    graphs, errors = [], []
    for s in raw:
        try:
            graphs.append(make_pyg_graph(s))
        except Exception as e:
            errors.append((s.get("sample_id"), str(e)))

    if errors:
        print(f"\nWARNING: {len(errors)} samples failed conversion:")
        for item in errors[:10]:
            print(" ", item)
        # SimNet data is cheap to regenerate; drops are logged but a
        # large failure rate indicates a generator bug:
        if len(errors) > 0.05 * len(raw):
            raise RuntimeError(
                f"{len(errors)}/{len(raw)} conversions failed (>5%) — "
                f"fix the generator instead of dropping."
            )

    train, val, test = split_dataset(graphs)
    stats = compute_stats(train)

    torch.save(graphs, OUT_FULL)
    torch.save(train, OUT_TRAIN)
    torch.save(val, OUT_VAL)
    torch.save(test, OUT_TEST)
    torch.save(stats, OUT_STATS)

    summarize("FULL", graphs)
    summarize("TRAIN", train)
    summarize("VAL", val)
    summarize("TEST", test)

    print("\nStats:")
    for k, v in stats.items():
        if not isinstance(v, (list, str)):
            print(f"  {k}: {v}")

    print("\nSaved:")
    for p in (OUT_FULL, OUT_TRAIN, OUT_VAL, OUT_TEST, OUT_STATS):
        print(" ", p)


if __name__ == "__main__":
    main()
