"""
Canonical Agent dataset builder (TE-CCL + ILP teacher -> unified schema).

Fixes vs. previous version
--------------------------
1. GROUP-BASED SPLITS (critical): the old random split leaked near-duplicate
   samples (same topology+variant, different message size) across
   train/val/test, so test metrics measured memorization, not
   generalization. We now split at the GROUP level with two modes:
     - "topology_holdout": entire topology families held out for val/test
       (tests generalization to unseen STRUCTURE)
     - "variant_holdout": link-parameter variants held out
       (tests generalization to unseen NETWORK PARAMETERS —
        exactly the thesis motivation)
2. FEATURES: per-graph relative normalization erased absolute physical
   scale (a 25 GBps uniform topology and a 200 GBps uniform topology
   looked identical). We keep relative features AND add log-scale absolute
   capacity/latency plus a physics-informed per-edge transmission-cost
   feature log10(message_size / capacity), i.e. the alpha-beta "beta*L"
   term the solver actually optimizes over.
3. STRUCTURAL ENCODING: random-walk structural encodings (RWSE) added to
   node features. Permutation-invariant, no node IDs, and known to help
   size generalization — the core "transfer across scales" claim.
4. SCHEDULING TARGET: the old target confused "used at epoch 0" with
   "never used" (both 0.0). We now (a) shift epochs by +1 so every used
   edge has target > 0, (b) normalize by completion_epoch (a meaningful,
   graph-level quantity) instead of the per-graph max, (c) export an
   explicit scheduling_mask so the loss can be computed on used edges
   only. The normalization factor is stored for inverse-transform.
5. TEACHER FLAGS REMOVED FROM u BY DEFAULT: telling the model which
   teacher produced the label invites shortcut learning and is undefined
   at inference time on new topologies. Kept behind a flag for an
   explicit "teacher conditioning" ablation.
6. RAW SCHEDULE PRESERVED: needed later for decoding, SimNet training,
   and error analysis; regenerating it costs ~300 s of ILP per sample.
7. TEACHER QUALITY carried through (mip_gap, wall time) for label
   filtering / weighting and honest speedup reporting.
"""

from pathlib import Path
from collections import Counter, defaultdict
import random
import math

import torch

from load_priors import compute_sp_load_priors
from tier_features import compute_capacity_classes, compute_role_bits


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"

TECCL_DATASET = DATA_DIR / "TE_CCL_dataset.pt"
ILP_DATASET = DATA_DIR / "allgather_teacher_dataset_incremental.pt"
SYCCL_DATASET = DATA_DIR / "syccl_dataset.pt"
SWITCHED_TOPO_DATASET = DATA_DIR / "switched_topo_teacher_dataset.pt"

OUT_FULL = DATA_DIR / "canonical_agent_dataset.pt"
OUT_TRAIN = DATA_DIR / "canonical_agent_train.pt"
OUT_VAL = DATA_DIR / "canonical_agent_val.pt"
OUT_TEST = DATA_DIR / "canonical_agent_test.pt"
OUT_STATS = DATA_DIR / "canonical_agent_stats.pt"

SEED = 42

# ------------------------------------------------------------
# Split configuration
# ------------------------------------------------------------
# "topology_holdout": hold out whole topology families (unseen structure)
# "variant_holdout" : hold out link-parameter variants  (unseen parameters)
SPLIT_MODE = "topology_holdout"

# Explicit holdouts (recommended for a thesis: fixed, documented, citable).
# If left empty, families are assigned deterministically from SEED.
# TEST spans two teachers on purpose (ilp_teacher + teccl) -- a
# same-domain-only test set can't measure cross-teacher generalization
# and lets a degenerate label (see load_edge_target) dominate the
# metric unchecked. "DGX2" was stale: this session's TE-CCL extraction
# renamed that topology to "DGX2_2_chassis", so it silently matched
# nothing and test ended up 100% ilp_teacher (ring_cluster only).
# SyCCL stays entirely in train on purpose: only 2 (now more, see
# clos16/clos32) topology families exist for it -- still folded
# entirely into train, no held-out SyCCL family yet.
#
# ilp_switched (parametric switched-family generator, switched_
# topologies.py): 18 families x 7 sizes, added specifically to test
# "generalizes to an UNSEEN switch family" -- the failure mode
# diagnosed via the DGX2_2_chassis holdout (r=0.03 on switch edges,
# 33x downstream CCT gap) and confirmed (via 3 independent negative
# interventions: switch-upweighting, SP-prior residual, more real
# SyCCL switch data) to be a family-generalization gap, not fixable by
# reweighting/priors/more-of-the-same-family data. These 3 holdouts
# are each a different template (star family is NOT held out at all,
# so the model sees plenty of star/single-switch topologies in train;
# these three test the OTHER three templates it must generalize to):
#   switched_dual_star_n12_r4    -- dual-switch, load-splitting
#   switched_two_tier_l4x4_r8    -- 2-tier hierarchy (leaf+spine)
#   switched_swclust_2x8_r4      -- clustered-mesh + 1 switch,
#                                    structurally closest analogue to
#                                    the real DGX2_2_chassis holdout
TEST_TOPOLOGIES = ["ring_cluster", "DGX2_2_chassis",
                   "switched_dual_star_n12_r4",
                   "switched_two_tier_l4x4_r8",
                   "switched_swclust_2x8_r4"]
VAL_TOPOLOGIES = ["two_bridge"]

# For variant_holdout mode: which variant_ids go where.
TEST_VARIANT_IDS = {8, 9}
VAL_VARIANT_IDS = {7}

# Teacher-conditioning ablation flag (default OFF — see fix #5).
INCLUDE_TEACHER_FLAGS = False

# Random-walk structural encoding depth.
RWSE_K = 8

# Drop samples whose teacher label is provably far from optimal.
# None = keep everything (gap is still stored for weighting/analysis).
MAX_MIP_GAP = None


# ============================================================
# Feature labels
# ============================================================

NODE_FEATURE_NAMES = (
    ["is_source", "degree_norm"]
    + [f"rwse_{k}" for k in range(1, RWSE_K + 1)]
)
# is_source replaces the old is_gpu/is_switch pair. "Switch" conflated
# two unrelated axes: structural identity (a physical switch's low
# degree / distinct capacity class is already visible to degree_norm
# + RWSE + edge capacity features — a redundant bit) and functional
# role (does this node originate a chunk, or only relay?). TE-CCL's
# NDv2 marks a fully-connected, ordinary-bandwidth GPU as a "switch"
# purely in its demand model (see get_source_indices) while our own
# ILP's switch is a structurally sparse, distinct-capacity physical
# node — one is_switch bit would mean two different things per
# teacher, a silent shortcut-learning risk. is_source carries only
# the functional role, identically defined across all teachers.

EDGE_FEATURE_NAMES = [
    "capacity_rel",          # per-graph relative (kept: helps transfer)
    "latency_rel",
    "log10_capacity",        # absolute physical scale (new)
    "log10_latency",
    "log10_tx_cost",         # log10(message_size / capacity): beta*L term
    "is_switch_edge",
    "is_inter_chassis_or_bridge",
    # NOTE: log1p_sp_load_latency / log1p_sp_load_hops (see
    # load_priors.py) were tried as extra edge FEATURES + a residual
    # training target and reverted -- downstream CCT-gap got WORSE
    # (overall 24.71x -> 48.41x) despite better edge-level correlation
    # (Pearson 0.603 -> 0.690). See train_agent.py's LOAD_TARGET
    # comment for the full writeup. sp_load_latency is still computed
    # here (below) and stored as load_residual_target /
    # sp_load_latency_log1p in the canonical sample for anyone who
    # wants to re-run that experiment, but is NOT appended to edge_attr
    # by default, so the model's input dimension matches the reported
    # baseline checkpoints.
    "capacity_class_norm",   # tier_features.py -- topology-general
    "is_top_class",          # "dimension vocabulary": which speed
                             # tier this edge belongs to (0=fastest),
                             # so a tier-1-relay pattern learned on
                             # NDv2/clos/switched-families can transfer
                             # to DGX2's own tier-1 edges even though
                             # absolute capacities never line up across
                             # families. Added after 4 independent data
                             # /reweighting interventions (switch
                             # upweight, SP-prior residual, real SyCCL
                             # expansion, 18 synthetic switch families)
                             # all failed to close the DGX2 gap --
                             # this is a representation-level try
                             # before going architectural.
    "src_is_source",         # edge role bits (also tier_features.py):
    "dst_is_source",         # GPU->GPU / GPU->relay / relay->GPU /
                             # relay->relay, free from existing
                             # is_source, no routing assumption.
]

GLOBAL_FEATURE_NAMES = [
    "log10_message_size",
    "log10_num_nodes",
    "is_allgather",
    "is_alltoall",
]
if INCLUDE_TEACHER_FLAGS:
    GLOBAL_FEATURE_NAMES += ["is_teccl", "is_ilp_teacher", "is_syccl"]

TARGET_NAMES = {
    "routing_edge_target":
        "binary edge usage target from teacher schedule -- DIAGNOSTIC "
        "ONLY: routing_positive_ratio is ~0.99-1.00 across all three "
        "teachers on these topologies (AllGather touches nearly every "
        "edge at least once), so this alone is close to uninformative "
        "as a training target. Use load_edge_target instead.",
    "load_edge_target":
        "log1p(count/amount of chunk-transfers over this edge) -- the "
        "informative routing-side target. 0 exactly where the edge is "
        "unused (log1p(0)=0), so it subsumes routing_edge_target: no "
        "separate binary head or mask needed. Regress with MSE/Huber "
        "over ALL edges.",
    "load_residual_target":
        "load_edge_target - sp_load_latency_log1p (see load_priors.py). "
        "Correction over the analytic shortest-path-load baseline; "
        "recover absolute load via pred + sp_load_latency_log1p. Added "
        "after switch-heavy OOD topologies (DGX2_2_chassis) showed the "
        "learned load_edge_target head has near-zero correlation there "
        "(r=0.03) while the analytic baseline alone reaches r=0.475 on "
        "the same holdout -- training on the residual lets the model "
        "correct the baseline instead of replacing it from scratch.",
    "sp_load_latency_log1p":
        "log1p(analytic demand-weighted shortest-path edge load, "
        "latency-weighted routing) -- NOT a training target on its own, "
        "the baseline load_residual_target is defined against.",
    "scheduling_edge_target":
        "(first_use_epoch + 1) / (norm_factor + 1) for used edges, 0 else",
    "scheduling_mask":
        "1.0 where routing target is positive; scheduling loss MUST be "
        "computed only on masked (used) edges",
    "scheduling_norm_factor":
        "graph-level normalization constant (completion_epoch when "
        "available, else max used epoch); needed for inverse transform",
}


# ============================================================
# Load utilities
# ============================================================

def load_list(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    data = torch.load(path, weights_only=False)
    if not isinstance(data, list):
        raise TypeError(f"Expected list in {path}, got {type(data)}")
    return data


def safe_float_message_size(x):
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).upper().strip()
    for suffix, mult in [("KB", 1024), ("MB", 1024 ** 2),
                         ("GB", 1024 ** 3), ("B", 1)]:
        if s.endswith(suffix):
            return float(s.replace(suffix, "")) * mult
    return float(s)


# ============================================================
# Switch / chassis metadata (unchanged logic)
# ============================================================

def get_switch_indices(sample, source_domain):
    if sample.get("switch_indices") is not None:
        return set(int(x) for x in sample["switch_indices"])

    topo = sample.get("topology_name", "")

    if source_domain == "teccl" and topo == "DGX2":
        return {0, 17}

    if source_domain == "teccl" and topo == "NDv2":
        n = sample["node_feat"].shape[0]
        if n == 33:
            return {0}
        return set()

    return set()


def get_source_indices(sample, num_nodes, switch_indices):
    """
    Functional role, not structure: which nodes actually originate a
    chunk in the schedule. Ground truth (sample["source_indices"],
    derived from event chunk_origin in teacher_common.derive_source_
    indices) is used whenever the adapter populated it; this is what
    caught TE-CCL's NDv2 node 0 being a fully-connected GPU that is
    nonetheless demand-model-relay-only.

    Falls back to (all nodes - switch_indices) only for adapters that
    don't yet carry chunk_origin (e.g. SyCCL's current dict route) —
    loud, not silent: any mismatch between the two when both are
    available is a real adapter/topology bug and must not pass
    quietly into a training label.
    """
    declared = sample.get("source_indices")
    if declared is not None:
        source_set = set(int(x) for x in declared)
        expected = set(range(num_nodes)) - set(switch_indices)
        if source_set != expected:
            raise ValueError(
                f"{sample.get('topology_name')}: schedule-derived "
                f"source_indices {sorted(source_set)} != "
                f"switch-complement expectation {sorted(expected)}. "
                f"switch_indices declared as {sorted(switch_indices)} "
                f"but the schedule disagrees — resolve before this "
                f"becomes a silently wrong label."
            )
        return source_set
    return set(range(num_nodes)) - set(switch_indices)


def infer_chassis_id(node_id, topology_name, num_nodes):
    if topology_name == "DGX2":
        return 0 if node_id <= 16 else 1
    if topology_name == "NDv2":
        if num_nodes == 16:
            return 0 if node_id < 8 else 1
        if num_nodes == 33:
            if node_id == 0:
                return -1
            return (node_id - 1) // 8
    return 0


# ============================================================
# Structural encoding (RWSE)
# ============================================================

def compute_rwse(num_nodes, edge_index, k_max=RWSE_K):
    """
    Random-walk structural encoding: diag(P^k) for k = 1..k_max,
    where P = D^{-1} A. Permutation-invariant, ID-free, and a standard
    remedy for GNN size-generalization. Dense math is fine at these
    graph sizes (< ~100 nodes).
    """
    A = torch.zeros(num_nodes, num_nodes)
    for u, v in edge_index.t().tolist():
        A[u, v] = 1.0
        A[v, u] = 1.0  # structural walk on the undirected skeleton

    deg = A.sum(dim=1).clamp(min=1.0)
    P = A / deg.unsqueeze(1)

    rwse = torch.zeros(num_nodes, k_max)
    Pk = torch.eye(num_nodes)
    for k in range(k_max):
        Pk = Pk @ P
        rwse[:, k] = torch.diagonal(Pk)

    return rwse


# ============================================================
# Node features
# ============================================================

def build_canonical_node_feat(num_nodes, edge_index, source_indices):
    degree = torch.zeros(num_nodes)
    for u, v in edge_index.t().tolist():
        degree[u] += 1.0
        degree[v] += 1.0

    max_degree = degree.max().item()
    if max_degree > 0:
        degree = degree / max_degree

    base = []
    for i in range(num_nodes):
        is_source = 1.0 if i in source_indices else 0.0
        base.append([is_source, float(degree[i])])
    base = torch.tensor(base, dtype=torch.float)

    rwse = compute_rwse(num_nodes, edge_index)

    return torch.cat([base, rwse], dim=1)


# ============================================================
# Edge features
# ============================================================

def _safe_log10(x: torch.Tensor) -> torch.Tensor:
    """log10 with clamping; zero/negative -> 0.0 sentinel (logged separately
    as relative feature anyway)."""
    out = torch.zeros_like(x)
    pos = x > 0
    out[pos] = torch.log10(x[pos])
    return out


def build_canonical_edge_attr(sample, switch_indices, message_size_bytes,
                              source_indices):
    edge_index = sample["edge_index"].long()
    old = sample["edge_attr"].float()
    if old.dim() == 1:
        old = old.view(-1, 1)

    num_edges = edge_index.shape[1]
    num_nodes = int(sample["node_feat"].shape[0])
    topology_name = sample.get("topology_name", "")

    capacity = old[:, 0].clone()
    latency = old[:, 1].clone() if old.shape[1] >= 2 else torch.zeros(num_edges)

    # Fail loudly on NaN/inf instead of silently zeroing (data bugs must
    # surface here, not three stages downstream in a loss curve).
    if not torch.isfinite(capacity).all():
        raise ValueError(
            f"Non-finite capacity in {topology_name}: "
            f"{capacity[~torch.isfinite(capacity)]}"
        )
    if not torch.isfinite(latency).all():
        raise ValueError(f"Non-finite latency in {topology_name}")

    # ---- relative (per-graph) ----
    capacity_rel = capacity / capacity.max() if capacity.max() > 0 else capacity
    latency_rel = latency / latency.max() if latency.max() > 0 else latency

    # ---- absolute (log scale) ----
    log_capacity = _safe_log10(capacity)
    log_latency = _safe_log10(latency)

    # ---- physics-informed: per-chunk transmission cost on this edge ----
    tx_cost = torch.zeros(num_edges)
    pos = capacity > 0
    tx_cost[pos] = torch.log10(
        torch.tensor(float(message_size_bytes)) / capacity[pos]
    )

    # ---- structural flags ----
    switch_edge, inter_or_bridge = [], []
    for u, v in edge_index.t().tolist():
        switch_edge.append(
            float((u in switch_indices) or (v in switch_indices))
        )
        cu = infer_chassis_id(u, topology_name, num_nodes)
        cv = infer_chassis_id(v, topology_name, num_nodes)
        inter_or_bridge.append(float(cu != cv and cu >= 0 and cv >= 0))

    # ---- analytic, topology-general load prior (see load_priors.py) ----
    sp_load_latency, sp_load_hops, prior_stats = compute_sp_load_priors(
        edge_index, latency, num_nodes, source_indices
    )
    if (prior_stats["unreachable_pairs_latency"] > 0
            or prior_stats["unreachable_pairs_hops"] > 0):
        raise ValueError(
            f"Unreachable AllGather demand pairs in {topology_name}: "
            f"{prior_stats} -- topology extraction is broken upstream."
        )

    # sp_load_hops is computed for the diagnostic/future-work path
    # (see load_priors.py) but currently unused; keep the variable name
    # referenced so linters don't flag it as dead computation.
    del sp_load_hops

    # ---- tier / role vocabulary (see tier_features.py) ----
    capacity_class_norm, is_top_class = compute_capacity_classes(capacity)
    src_is_source, dst_is_source = compute_role_bits(
        edge_index, source_indices, num_nodes)

    return torch.stack(
        [
            capacity_rel.float(),
            latency_rel.float(),
            log_capacity.float(),
            log_latency.float(),
            tx_cost.float(),
            torch.tensor(switch_edge, dtype=torch.float),
            torch.tensor(inter_or_bridge, dtype=torch.float),
            capacity_class_norm.float(),
            is_top_class.float(),
            src_is_source.float(),
            dst_is_source.float(),
        ],
        dim=1,
    ), sp_load_latency


# ============================================================
# Targets
# ============================================================

def matrix_target_to_edge_target(matrix_target, edge_index, default=0.0):
    matrix_target = matrix_target.float()
    values = []
    for u, v in edge_index.t().tolist():
        if u < matrix_target.shape[0] and v < matrix_target.shape[1]:
            values.append(float(matrix_target[u, v]))
        else:
            values.append(default)
    return torch.tensor(values, dtype=torch.float)


def build_scheduling_target(raw_epochs, routing_target, completion_epoch):
    """
    Target design (fix #4):
      - used edges:   (first_use_epoch + 1) / (norm + 1)  in (0, 1]
      - unused edges: 0.0, EXCLUDED from the loss via scheduling_mask
    The +1 shift removes the 'epoch 0 vs never used' ambiguity.
    Normalizing by completion_epoch (not per-graph max first-use) makes
    the target a physically meaningful fraction of the collective's
    lifetime and gives a well-defined inverse transform at inference.
    """
    y = raw_epochs.clone().float()
    mask = (routing_target > 0).float()
    used = mask.bool()

    if completion_epoch is not None and completion_epoch > 0:
        norm = float(completion_epoch)
    elif used.any():
        norm = float(y[used].max().item())
    else:
        norm = 1.0

    y_out = torch.zeros_like(y)
    if used.any():
        y_out[used] = (y[used] + 1.0) / (norm + 1.0)

    return y_out, mask, norm


# ============================================================
# Canonicalization
# ============================================================

def canonicalize_sample(sample, source_domain):
    sample = dict(sample)

    required = ["topology_name", "message_size", "node_feat",
                "edge_index", "edge_attr", "routing_target",
                "load_target", "scheduling_target"]
    missing = [k for k in required if k not in sample]
    if missing:
        raise KeyError(f"Missing keys in {source_domain}: {missing}")

    edge_index = sample["edge_index"].long()
    num_nodes = int(sample["node_feat"].shape[0])
    message_size_bytes = safe_float_message_size(sample["message_size"])
    switch_indices = get_switch_indices(sample, source_domain)
    source_indices = get_source_indices(sample, num_nodes, switch_indices)
    completion_epoch = sample.get("completion_epoch", None)

    # ---- features ----
    node_feat = build_canonical_node_feat(num_nodes, edge_index,
                                          source_indices)
    edge_attr, sp_load_latency = build_canonical_edge_attr(
        sample, switch_indices, message_size_bytes, source_indices
    )

    # ---- targets ----
    routing_edge_target = (
        matrix_target_to_edge_target(sample["routing_target"], edge_index)
        > 0
    ).float()

    # load_edge_target is the primary regression target (see
    # TARGET_NAMES): log1p of per-edge chunk-transfer count/amount.
    # log1p(0) == 0, so unused edges fall out naturally -- no mask
    # needed for this head, unlike the old binary routing head.
    raw_load = matrix_target_to_edge_target(sample["load_target"], edge_index)
    load_edge_target = torch.log1p(raw_load.clamp(min=0.0))

    # residual target (see load_priors.py): correction over the
    # analytic shortest-path-load baseline, so an OOD prediction
    # degrades to "baseline + noise" instead of a blind guess.
    sp_load_log1p = torch.log1p(sp_load_latency.float())
    load_residual_target = load_edge_target - sp_load_log1p

    raw_sched = matrix_target_to_edge_target(
        sample["scheduling_target"], edge_index
    )
    scheduling_edge_target, scheduling_mask, norm_factor = \
        build_scheduling_target(raw_sched, routing_edge_target,
                                completion_epoch)

    # ---- global features ----
    collective = sample.get("collective", "AllGather")
    global_vals = [
        math.log10(max(1.0, message_size_bytes)),
        math.log10(max(1.0, float(num_nodes))),
        float(collective == "AllGather"),
        float(collective == "AlltoAll"),
    ]
    if INCLUDE_TEACHER_FLAGS:
        global_vals += [
            float(source_domain == "teccl"),
            float(source_domain == "ilp_teacher"),
            float(source_domain == "syccl"),
        ]
    global_u = torch.tensor(global_vals, dtype=torch.float)

    # ---- group key: the atomic unit for splitting ----
    group_key = f"{source_domain}::{sample.get('topology_name')}"

    out = {
        "source_domain": source_domain,
        "source": sample.get("source", source_domain),
        "topology_name": sample.get("topology_name"),
        "collective": collective,
        "mode": sample.get(
            "mode",
            {"ilp_teacher": "ILP", "ilp_switched": "ILP",
             "teccl": "TECCL",
             "syccl": "SYCCL"}.get(source_domain, source_domain),
        ),
        "chassis": sample.get("chassis"),
        "variant_id": sample.get("variant_id"),
        "link_variant_seed": sample.get("link_variant_seed"),
        "group_key": group_key,

        "message_size_raw": sample.get("message_size"),
        "message_size_bytes": message_size_bytes,
        "num_nodes": num_nodes,
        "num_edges": int(edge_index.shape[1]),
        "switch_indices": sorted(switch_indices),

        # graph tensors
        "node_feat": node_feat,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "u": global_u,

        # targets
        "routing_edge_target": routing_edge_target,
        "load_edge_target": load_edge_target,
        "load_residual_target": load_residual_target,
        "sp_load_latency_log1p": sp_load_log1p,
        "scheduling_edge_target": scheduling_edge_target,
        "scheduling_mask": scheduling_mask,
        "scheduling_norm_factor": norm_factor,

        # teacher quality (fix #7)
        "mip_gap": sample.get("mip_gap"),
        "teacher_wall_time_sec": sample.get("solver_wall_time_sec"),
        "solver_status": sample.get("solver_status"),

        # labels
        "node_feature_names": NODE_FEATURE_NAMES,
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "global_feature_names": GLOBAL_FEATURE_NAMES,
        "target_names": TARGET_NAMES,
    }

    # ---- raw schedule preserved (fix #6) ----
    if "schedule" in sample:
        out["raw_schedule"] = sample["schedule"]

    for k in ("completion_epoch", "completion_time", "time_target"):
        if k in sample:
            out[k] = sample[k]

    return out


# ============================================================
# Group-based splitting (fix #1)
# ============================================================

def split_by_topology(data):
    """Whole topology families go to exactly one split."""
    groups = defaultdict(list)
    for s in data:
        groups[s["topology_name"]].append(s)

    all_topos = sorted(groups.keys())

    test_topos = [t for t in TEST_TOPOLOGIES if t in groups]
    val_topos = [t for t in VAL_TOPOLOGIES if t in groups]

    # Deterministic fallback if config lists are empty/stale.
    if not test_topos or not val_topos:
        rng = random.Random(SEED)
        shuffled = list(all_topos)
        rng.shuffle(shuffled)
        if not test_topos:
            test_topos = shuffled[:max(1, len(shuffled) // 5)]
        if not val_topos:
            remaining = [t for t in shuffled if t not in test_topos]
            val_topos = remaining[:max(1, len(remaining) // 5)]

    train, val, test = [], [], []
    for topo, items in groups.items():
        if topo in test_topos:
            test.extend(items)
        elif topo in val_topos:
            val.extend(items)
        else:
            train.extend(items)

    print(f"\n[split: topology_holdout]")
    print(f"  train topologies: "
          f"{sorted(set(all_topos) - set(test_topos) - set(val_topos))}")
    print(f"  val topologies:   {sorted(val_topos)}")
    print(f"  test topologies:  {sorted(test_topos)}")

    return train, val, test


def split_by_variant(data):
    """Link-parameter variants held out; structure seen in training."""
    train, val, test = [], [], []
    for s in data:
        vid = s.get("variant_id")
        if vid is None:
            # Samples without variants (e.g. TE-CCL) stay in train:
            # this split mode measures parameter generalization on the
            # ILP families only.
            train.append(s)
        elif vid in TEST_VARIANT_IDS:
            test.append(s)
        elif vid in VAL_VARIANT_IDS:
            val.append(s)
        else:
            train.append(s)

    print(f"\n[split: variant_holdout]")
    print(f"  val variants:  {sorted(VAL_VARIANT_IDS)}")
    print(f"  test variants: {sorted(TEST_VARIANT_IDS)}")

    return train, val, test


def split_dataset(data):
    if SPLIT_MODE == "topology_holdout":
        return split_by_topology(data)
    if SPLIT_MODE == "variant_holdout":
        return split_by_variant(data)
    raise ValueError(f"Unknown SPLIT_MODE: {SPLIT_MODE}")


def assert_no_group_leakage(train, val, test):
    """Hard guarantee: no group key appears in more than one split."""
    if SPLIT_MODE != "topology_holdout":
        return
    tr = {s["group_key"] for s in train}
    va = {s["group_key"] for s in val}
    te = {s["group_key"] for s in test}
    assert not (tr & va), f"train/val leakage: {tr & va}"
    assert not (tr & te), f"train/test leakage: {tr & te}"
    assert not (va & te), f"val/test leakage: {va & te}"
    print("  leakage check: PASSED (group keys disjoint)")


# ============================================================
# Dataset statistics (incl. pos_weight for class imbalance)
# ============================================================

def compute_stats(train):
    y = torch.cat([s["routing_edge_target"] for s in train])
    n_pos = float(y.sum().item())
    n_neg = float(y.numel() - n_pos)
    pos_weight = n_neg / max(1.0, n_pos)

    masks = torch.cat([s["scheduling_mask"] for s in train])
    scheds = torch.cat([s["scheduling_edge_target"] for s in train])
    used = masks.bool()

    loads = torch.cat([s["load_edge_target"] for s in train])
    residuals = torch.cat([s["load_residual_target"] for s in train])

    stats = {
        # routing_* kept for diagnostics; routing_positive_ratio is
        # ~0.99-1.00 (see load_* below for the informative target)
        "routing_positive_ratio": n_pos / max(1.0, y.numel()),
        "routing_pos_weight": pos_weight,          # feed to BCEWithLogitsLoss
        "load_mean": float(loads.mean()),
        "load_std": float(loads.std()),
        "load_used_mean": float(loads[used].mean()) if used.any() else 0.0,
        "load_used_std": float(loads[used].std()) if used.any() else 0.0,
        "residual_mean": float(residuals.mean()),
        "residual_std": float(residuals.std()),
        "scheduling_used_mean": float(scheds[used].mean()) if used.any() else 0.0,
        "scheduling_used_std": float(scheds[used].std()) if used.any() else 0.0,
        "num_train": len(train),
        "split_mode": SPLIT_MODE,
        "seed": SEED,
        "rwse_k": RWSE_K,
        "include_teacher_flags": INCLUDE_TEACHER_FLAGS,
        "node_feature_names": NODE_FEATURE_NAMES,
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "global_feature_names": GLOBAL_FEATURE_NAMES,
    }
    return stats


# ============================================================
# Summaries / checks
# ============================================================

def check_dataset(data):
    print("\nCanonical dataset check")
    print("-" * 70)
    node_dims = Counter(int(s["node_feat"].shape[1]) for s in data)
    edge_dims = Counter(int(s["edge_attr"].shape[1]) for s in data)
    global_dims = Counter(int(s["u"].shape[0]) for s in data)

    for s in data:
        ne = s["edge_index"].shape[1]
        assert s["edge_attr"].shape[0] == ne
        assert s["routing_edge_target"].shape[0] == ne
        assert s["load_edge_target"].shape[0] == ne
        assert s["load_residual_target"].shape[0] == ne
        assert s["sp_load_latency_log1p"].shape[0] == ne
        assert torch.allclose(
            s["load_edge_target"] - s["sp_load_latency_log1p"],
            s["load_residual_target"], atol=1e-5
        ), f"residual != load - baseline in {s['topology_name']}"
        assert s["scheduling_edge_target"].shape[0] == ne
        assert s["scheduling_mask"].shape[0] == ne
        # mask consistency: scheduling target strictly positive iff used
        used = s["scheduling_mask"].bool()
        assert (s["scheduling_edge_target"][used] > 0).all(), \
            f"zero scheduling target on used edge in {s['topology_name']}"
        assert (s["scheduling_edge_target"][~used] == 0).all()
        # load_edge_target: 0 exactly where unused, >0 exactly where
        # used -- it must agree with the same mask, since it subsumes
        # the old binary routing signal.
        assert (s["load_edge_target"][used] > 0).all(), \
            f"zero load target on used edge in {s['topology_name']}"
        assert (s["load_edge_target"][~used] == 0).all(), \
            f"nonzero load target on unused edge in {s['topology_name']}"

    print("node_feat dims:", node_dims)
    print("edge_attr dims:", edge_dims)
    print("global dims:", global_dims)


def summarize(name, data):
    print(f"\n{name}")
    print("-" * 70)
    print("count:", len(data))
    print("source_domain:", Counter(s["source_domain"] for s in data))
    print("topologies:", Counter(s["topology_name"] for s in data))
    print("collectives:", Counter(s["collective"] for s in data))
    print("node_counts:", Counter(s["num_nodes"] for s in data))
    gaps = [s["mip_gap"] for s in data if s.get("mip_gap") is not None]
    if gaps:
        print("mip_gap: n=%d, mean=%.4f, max=%.4f"
              % (len(gaps), sum(gaps) / len(gaps), max(gaps)))


# ============================================================
# Main
# ============================================================

def main():
    teccl_raw = load_list(TECCL_DATASET)
    ilp_raw = load_list(ILP_DATASET)
    sources = [("teccl", teccl_raw), ("ilp_teacher", ilp_raw)]

    if SYCCL_DATASET.exists():
        sources.append(("syccl", load_list(SYCCL_DATASET)))
    else:
        print(f"NOTE: {SYCCL_DATASET.name} not found, skipping SyCCL "
              f"(run src/data/extract_syccl.py to generate it).")

    if SWITCHED_TOPO_DATASET.exists():
        sources.append(("ilp_switched", load_list(SWITCHED_TOPO_DATASET)))
    else:
        print(f"NOTE: {SWITCHED_TOPO_DATASET.name} not found, skipping "
              f"switched-topology families (run "
              f"src/data/generate_switched_teacher_dataset.py).")

    clean, skipped = [], []

    for source_domain, raw in sources:
        for s in raw:
            try:
                c = canonicalize_sample(s, source_domain)
                if (MAX_MIP_GAP is not None
                        and c.get("mip_gap") is not None
                        and c["mip_gap"] > MAX_MIP_GAP):
                    skipped.append((source_domain,
                                    c["topology_name"],
                                    f"mip_gap {c['mip_gap']} > {MAX_MIP_GAP}"))
                    continue
                clean.append(c)
            except Exception as e:
                skipped.append((source_domain,
                                s.get("topology_name"), str(e)))

    if not clean:
        raise RuntimeError("No valid samples produced.")

    check_dataset(clean)

    train, val, test = split_dataset(clean)
    assert_no_group_leakage(train, val, test)

    stats = compute_stats(train)

    torch.save(clean, OUT_FULL)
    torch.save(train, OUT_TRAIN)
    torch.save(val, OUT_VAL)
    torch.save(test, OUT_TEST)
    torch.save(stats, OUT_STATS)

    print("\n" + "=" * 70)
    print("DONE")
    print("\nSaved:", OUT_FULL, OUT_TRAIN, OUT_VAL, OUT_TEST, OUT_STATS,
          sep="\n  ")

    summarize("FULL", clean)
    summarize("TRAIN", train)
    summarize("VAL", val)
    summarize("TEST", test)

    print("\nTrain stats:")
    for k, v in stats.items():
        if not isinstance(v, list):
            print(f"  {k}: {v}")

    print("\nSkipped:", len(skipped))
    for item in skipped[:10]:
        print(" ", item)


if __name__ == "__main__":
    main()
