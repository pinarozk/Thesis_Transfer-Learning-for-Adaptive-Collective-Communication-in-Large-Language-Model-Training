"""
tier_features.py — a topology-general "dimension vocabulary" for the
canonical edge features.

WHY (and why this is NOT the SP-prior intervention again):
The SP load prior failed because it injected a ROUTING assumption --
it told the model where load should go, suppressing the learned
deviation that makes AllGather schedules good. These features inject
no routing at all. They only name what physically exists:

  capacity_class  Which speed TIER is this edge? Distinct capacity
                  values in a graph are ranked (fastest = 0). On real
                  fabrics tiers align with the SyCCL paper's
                  "dimensions": tier 0 ~ NVLink/intra-server, tier 1 ~
                  leaf uplinks, tier 2 ~ spine... A model that learns
                  "tier-1 relay edges carry aggregated cross-group
                  load" on NDv2/clos/switched-families can carry that
                  RELATION to DGX2's tier-1 edges -- the bridge that
                  is currently missing, because absolute log-
                  capacities of different families never line up.

  edge role bits  From is_source of the endpoints (already computed).
                  Together they separate GPU->GPU, GPU->relay
                  (fan-in), relay->GPU (fan-out), relay->relay (trunk)
                  -- four structurally different load regimes that
                  today share one head with no disambiguating input.

Both are pure computations on the topology -- no shortest paths, no
demand model, no assumption about where load ought to flow.

INTEGRATION (exact steps)
-------------------------
build_canonical_dataset.py:
  1. from src.data.tier_features import (compute_capacity_classes,
                                         compute_role_bits)
  2. EDGE_FEATURE_NAMES += ["capacity_class_norm", "is_top_class",
                            "src_is_source", "dst_is_source"]
  3. In build_canonical_edge_attr(), after capacity is known and
     source_indices is available: append the four columns.
  4. Rebuild canonical + pyg (raw schedules on disk; no re-solving).
"""

import torch


def compute_capacity_classes(capacity_raw, rel_tol=0.05):
    """
    Rank distinct capacity values within ONE graph into speed tiers.

    Args:
        capacity_raw: FloatTensor [E] -- absolute capacities (any unit,
                      consistent within the graph).
        rel_tol: values within 5% are the same tier (absorbs the
                 per-variant jitter of the generators).

    Returns:
        class_norm: FloatTensor [E] in [0, 1] -- tier / (n_tiers - 1),
                    0 = fastest tier, 1 = slowest. Normalized so the
                    encoding is comparable across graphs with
                    different tier counts (a 2-tier and a 4-tier
                    fabric both map fastest->0, slowest->1).
        is_top:     FloatTensor [E] -- 1 for the fastest tier (the
                    "intra-server dimension" indicator).
    """
    caps = capacity_raw.double()
    order = torch.argsort(caps, descending=True)
    tier_of = torch.zeros_like(caps, dtype=torch.long)
    tier = 0
    prev = None
    for idx in order.tolist():
        c = caps[idx].item()
        if prev is not None and (prev - c) / max(prev, 1e-30) > rel_tol:
            tier += 1
        tier_of[idx] = tier
        prev = c
    n_tiers = int(tier_of.max().item()) + 1
    denom = max(n_tiers - 1, 1)
    class_norm = (tier_of.float() / denom)
    is_top = (tier_of == 0).float()
    return class_norm, is_top


def compute_role_bits(edge_index, source_indices, num_nodes):
    """
    Per-edge endpoint roles from schedule-derived is_source.

    Returns (src_is_source[E], dst_is_source[E]) as float tensors.
    The 2-bit combination spans four regimes:
        (1,1) GPU->GPU        peer traffic
        (1,0) GPU->relay      fan-in onto the fabric
        (0,1) relay->GPU      fan-out from the fabric
        (0,0) relay->relay    trunk / inter-tier
    """
    is_src_node = torch.zeros(num_nodes)
    for i in source_indices:
        is_src_node[int(i)] = 1.0
    src_b = is_src_node[edge_index[0]]
    dst_b = is_src_node[edge_index[1]]
    return src_b, dst_b


# ============================================================
# One-shot sanity check on synthetic tiers
# ============================================================

if __name__ == "__main__":
    # 3-tier fabric: 50G mesh, 25G leaf, 12.5G spine (+jitter)
    caps = torch.tensor([50e9, 51e9, 49e9, 25e9, 24.5e9,
                         12.5e9, 12.4e9])
    cn, top = compute_capacity_classes(caps)
    print("class_norm:", cn.tolist())
    print("is_top    :", top.tolist())
    assert cn.min() == 0.0 and cn.max() == 1.0
    assert top.sum() == 3          # the three ~50G edges
    ei = torch.tensor([[0, 0, 4, 4], [1, 4, 0, 5]])
    sb, db = compute_role_bits(ei, source_indices=[0, 1], num_nodes=6)
    print("src_bits  :", sb.tolist())   # [1,1,0,0]
    print("dst_bits  :", db.tolist())   # [1,0,1,0]
    print("OK")
