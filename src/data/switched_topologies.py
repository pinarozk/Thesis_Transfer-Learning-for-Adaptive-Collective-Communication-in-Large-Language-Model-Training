"""
switched_topologies.py — parametric switched-family generator for the
ILP teacher.

WHY (ties directly to the measured failure):
The intra-DC target makes the switch regime the main event, and both
rejected interventions point to the same root cause: the model has
seen too FEW distinct switch families to generalize to an unseen one
(DGX2: 33x downstream gap). Reweighting cannot inject family
knowledge; SP priors inject the WRONG knowledge. The only legitimate
source is data: many structurally different switched families, cheap
enough for the own-ILP teacher to solve (8-16 GPUs).

This module contributes 4 parametric TEMPLATES x parameter sweeps
= 18 distinct switch families:

  star              1 switch, N GPUs                    (NVSwitch-like)
  dual_star         2 switches, GPUs dual-homed         (redundant fabric)
  two_tier          leaf switches + 1 spine             (mini leaf-spine)
  switched_clusters k full-mesh GPU clusters joined
                    only through a switch               (DGX2-chassis-like)

Design axes (why these parameters):
  * gpu/switch capacity ratio r in {2, 4, 8}: the SyCCL paper's 7:1 vs
    3.6:1 bandwidth-ratio discussion shows this ratio is what shapes
    optimal schedules on switched fabrics -- sweeping it forces the
    model to learn ratio-conditioned load allocation instead of one
    family's constant.
  * scale 8-16 GPUs: within the ILP solver's practical reach.
  * capacity/latency jitter per variant: same connectivity, different
    physics -- the thesis' "variable network parameters" axis, now on
    switched structure.

CONVENTIONS (matched to the existing generator):
  capacities in the same absolute scale as topo_generate.py's
  generate_delay_and_capacity (fastest_linkrate = 50e9), latencies in
  seconds (0.6-1.6 us). Switches appended AFTER GPU indices, physical
  links emitted as two directed (symmetric) entries. Deterministic via
  hashlib-derived seeds.

ILP COMPATIBILITY NOTE:
build_merged_dataset.solve_allgather_ilp treats switch_indices as
excluded source/destination nodes (via
generate_all_gather_single_chunk_with_switch), demand defined only
over source_indices (the GPU nodes) -- see the ADAPTER at the bottom.
"""

import hashlib
import random
from dataclasses import dataclass, field


def deterministic_seed(*parts) -> int:
    key = "|".join(str(p) for p in parts)
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 1_000_000


# ============================================================
# Output container
# ============================================================

@dataclass
class SwitchedTopo:
    name: str                    # e.g. "star_n8_r4"
    num_nodes: int               # GPUs + switches
    gpu_indices: list
    switch_indices: list
    capacity_bps: list           # dense [n][n], 0 = no link
    latency_s: list              # dense [n][n]
    params: dict = field(default_factory=dict)

    def edges(self):
        for u in range(self.num_nodes):
            for v in range(self.num_nodes):
                if u != v and self.capacity_bps[u][v] > 0:
                    yield u, v, self.capacity_bps[u][v], \
                        self.latency_s[u][v]

    def to_topo_demo(self):
        """0/1 adjacency (list of lists), the format solve_allgather_ilp
        expects for tree generation."""
        n = self.num_nodes
        adj = [[0] * n for _ in range(n)]
        for u, v, _, _ in self.edges():
            adj[u][v] = 1
        return adj


def _empty(n):
    return ([[0.0] * n for _ in range(n)],
            [[0.0] * n for _ in range(n)])


def _link(cap, lat, u, v, c, l):
    """bidirectional physical link -> two directed entries (possibly
    asymmetric later; symmetric here)."""
    cap[u][v] = cap[v][u] = float(c)
    lat[u][v] = lat[v][u] = float(l)


def _jitter(rng, base, spread=0.15):
    return base * (1.0 + rng.uniform(-spread, spread))


# ============================================================
# Family templates
# ============================================================

def build_star(n_gpus, ratio, seed):
    """1 switch, every GPU single-homed to it. All traffic crosses the
    switch: the purest switch-bottleneck family (NVSwitch abstraction).
    ratio = gpu_link_capacity / switch_uplink_capacity ... here there
    is only one link class (GPU<->switch), so ratio scales it against
    the 50e9 reference: link = 50e9 / ratio."""
    rng = random.Random(seed)
    n = n_gpus + 1
    sw = n_gpus
    cap, lat = _empty(n)
    link_c = 50e9 / ratio
    for g in range(n_gpus):
        _link(cap, lat, g, sw, _jitter(rng, link_c),
              _jitter(rng, 1.0e-6))
    return SwitchedTopo(
        name=f"star_n{n_gpus}_r{ratio}",
        num_nodes=n, gpu_indices=list(range(n_gpus)),
        switch_indices=[sw], capacity_bps=cap, latency_s=lat,
        params={"family": "star", "n_gpus": n_gpus, "ratio": ratio},
    )


def build_dual_star(n_gpus, ratio, seed):
    """2 switches, every GPU dual-homed. Load can be SPLIT across two
    relays -- teaches parallel-tree spreading across switch planes."""
    rng = random.Random(seed)
    n = n_gpus + 2
    s0, s1 = n_gpus, n_gpus + 1
    cap, lat = _empty(n)
    link_c = 50e9 / ratio
    for g in range(n_gpus):
        _link(cap, lat, g, s0, _jitter(rng, link_c),
              _jitter(rng, 1.0e-6))
        _link(cap, lat, g, s1, _jitter(rng, link_c),
              _jitter(rng, 1.0e-6))
    # inter-switch trunk (lets one plane relieve the other)
    _link(cap, lat, s0, s1, _jitter(rng, 2 * link_c),
          _jitter(rng, 1.3e-6))
    return SwitchedTopo(
        name=f"dual_star_n{n_gpus}_r{ratio}",
        num_nodes=n, gpu_indices=list(range(n_gpus)),
        switch_indices=[s0, s1], capacity_bps=cap, latency_s=lat,
        params={"family": "dual_star", "n_gpus": n_gpus,
                "ratio": ratio},
    )


def build_two_tier(n_leaves, gpus_per_leaf, ratio, seed):
    """Mini leaf-spine: each leaf switch serves a GPU group; leaves
    join through one spine. Two relay TIERS -- the hierarchical
    structure real Clos schedules exploit."""
    rng = random.Random(seed)
    n_gpus = n_leaves * gpus_per_leaf
    n = n_gpus + n_leaves + 1
    leaf0 = n_gpus
    spine = n_gpus + n_leaves
    cap, lat = _empty(n)
    gpu_c = 50e9 / ratio
    up_c = gpu_c * gpus_per_leaf / 2.0     # mild oversubscription
    for lf in range(n_leaves):
        sw = leaf0 + lf
        for j in range(gpus_per_leaf):
            g = lf * gpus_per_leaf + j
            _link(cap, lat, g, sw, _jitter(rng, gpu_c),
                  _jitter(rng, 0.8e-6))
        _link(cap, lat, sw, spine, _jitter(rng, up_c),
              _jitter(rng, 1.3e-6))
    return SwitchedTopo(
        name=f"two_tier_l{n_leaves}x{gpus_per_leaf}_r{ratio}",
        num_nodes=n, gpu_indices=list(range(n_gpus)),
        switch_indices=list(range(leaf0, n)),
        capacity_bps=cap, latency_s=lat,
        params={"family": "two_tier", "n_leaves": n_leaves,
                "gpus_per_leaf": gpus_per_leaf, "ratio": ratio},
    )


def build_switched_clusters(n_clusters, cluster_size, ratio, seed):
    """k full-mesh GPU clusters joined ONLY through a central switch --
    the DGX2-chassis abstraction: fast intra-cluster mesh, all
    cross-cluster traffic squeezed through one relay."""
    rng = random.Random(seed)
    n_gpus = n_clusters * cluster_size
    n = n_gpus + 1
    sw = n_gpus
    cap, lat = _empty(n)
    mesh_c = 50e9
    up_c = mesh_c / ratio
    for c in range(n_clusters):
        base = c * cluster_size
        for i in range(cluster_size):
            for j in range(i + 1, cluster_size):
                _link(cap, lat, base + i, base + j,
                      _jitter(rng, mesh_c), _jitter(rng, 0.6e-6))
        # one uplink per cluster head + one per tail: two attach
        # points so intra-cluster position matters
        _link(cap, lat, base, sw, _jitter(rng, up_c),
              _jitter(rng, 1.3e-6))
        _link(cap, lat, base + cluster_size - 1, sw,
              _jitter(rng, up_c), _jitter(rng, 1.3e-6))
    return SwitchedTopo(
        name=f"swclust_{n_clusters}x{cluster_size}_r{ratio}",
        num_nodes=n, gpu_indices=list(range(n_gpus)),
        switch_indices=[sw], capacity_bps=cap, latency_s=lat,
        params={"family": "switched_clusters",
                "n_clusters": n_clusters,
                "cluster_size": cluster_size, "ratio": ratio},
    )


# ============================================================
# The family grid (train on many, hold out a few)
# ============================================================

def family_grid():
    """
    Returns list of (family_name, zero-arg builder) pairs.
    18 distinct switched families. Suggested holdout at split time
    (closest structural relatives of real held-out fabrics):
        swclust_2x8_r4, two_tier_l4x4_r8, dual_star_n12_r4
    """
    grid = []
    for n_gpus in (8, 12, 16):
        for ratio in (2, 4, 8):
            seed = deterministic_seed("star", n_gpus, ratio)
            grid.append((f"star_n{n_gpus}_r{ratio}",
                         lambda n=n_gpus, r=ratio, s=seed:
                         build_star(n, r, s)))
    for n_gpus, ratio in ((8, 2), (12, 4), (16, 8)):
        seed = deterministic_seed("dual", n_gpus, ratio)
        grid.append((f"dual_star_n{n_gpus}_r{ratio}",
                     lambda n=n_gpus, r=ratio, s=seed:
                     build_dual_star(n, r, s)))
    for n_leaves, gpl, ratio in ((2, 4, 4), (4, 4, 8), (2, 8, 4)):
        seed = deterministic_seed("tier", n_leaves, gpl, ratio)
        grid.append((f"two_tier_l{n_leaves}x{gpl}_r{ratio}",
                     lambda a=n_leaves, b=gpl, r=ratio, s=seed:
                     build_two_tier(a, b, r, s)))
    for n_cl, cs, ratio in ((2, 4, 2), (2, 8, 4), (4, 4, 4)):
        seed = deterministic_seed("swcl", n_cl, cs, ratio)
        grid.append((f"swclust_{n_cl}x{cs}_r{ratio}",
                     lambda a=n_cl, b=cs, r=ratio, s=seed:
                     build_switched_clusters(a, b, r, s)))
    return grid


if __name__ == "__main__":
    for name, b in family_grid():
        t = b()
        e = sum(1 for _ in t.edges())
        print(f"{name:26s} nodes={t.num_nodes:3d} "
              f"gpus={len(t.gpu_indices):3d} "
              f"switches={len(t.switch_indices)} edges={e}")
