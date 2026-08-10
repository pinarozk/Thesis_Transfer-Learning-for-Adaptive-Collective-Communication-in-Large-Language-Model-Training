"""
teacher_common.py — one normalized schema for every teacher solver.

We now have THREE teachers (own Gurobi ILP, microsoft/TE-CCL,
aliyun/SyCCL), each with its own output format. Instead of teaching the
canonical builder three dialects, every adapter converts its solver's
output into ONE normal form here. The canonical builder then needs to
know nothing about any solver.

NORMAL FORM (dict per sample) — exactly what canonicalize_sample() in
build_canonical_dataset.py consumes:

  topology_name      str
  collective         "AllGather" | "AlltoAll" | ...
  message_size       float (bytes)  [chunk/shard size per source]
  node_feat          FloatTensor [N, 1] zeros — placeholder; the
                     canonical builder REBUILDS node features and only
                     reads shape[0] for N
  edge_index         LongTensor [2, E] directed edges
  edge_attr          FloatTensor [E, 2] = (capacity_raw, latency_raw)
  capacity_unit      str, e.g. "Gbps" | "GBps" | "chunks_per_epoch" —
                     recorded so CAPACITY_TO_BPS in schedule_bridge
                     stays the single conversion point, per source
  routing_target     FloatTensor [N, N]; >0 iff any transfer used (u,v)
                     (kept for diagnostics; ~0.99-1.00 positive across
                     all three teachers on these topologies, i.e.
                     close to uninformative on its own -- see
                     load_target)
  load_target        FloatTensor [N, N]; COUNT of chunk-transfers over
                     (u,v). The informative routing-side target: edge
                     load varies (bottleneck edges carry far more
                     traffic) even where routing_target is ~constant.
                     load_target>0 exactly where routing_target>0, so
                     one regression head over this replaces the
                     degenerate binary routing head.
  scheduling_target  FloatTensor [N, N]; first-use EPOCH of (u,v),
                     0 where unused (canonical builder handles the
                     epoch-0-vs-unused ambiguity via the mask)
  completion_epoch   int (max epoch over events)
  completion_time    float | None (seconds, if the solver reports it)
  schedule           list of normalized events (below) — kept raw for
                     SimNet featurization / decoding later
  switch_indices     list[int] — node ids EXPECTED not to originate a
                     chunk (physical switch OR a demand-model relay
                     role assigned to an ordinary node — these are NOT
                     the same thing; see source_indices)
  source_indices     list[int] | None — node ids that ACTUALLY
                     originate >=1 chunk in this schedule (derived
                     from event chunk_origin). This is the ground
                     truth for the "is_source" node feature; None
                     when the adapter doesn't populate chunk_origin
                     yet (canonical builder must fall back explicitly)
  source             str: "ilp_teacher" | "teccl" | "syccl"
  solver_wall_time_sec, mip_gap, solver_status — quality metadata
                     (None when unknown; NEVER guessed)
  epoch_duration_sec float | None — makes scheduling epochs
                     comparable across teachers when known

NORMALIZED EVENT:
  {"chunk": str, "src": int, "dst": int, "epoch": int}
One event = one chunk crossing one directed edge in one epoch.
Multi-hop transfers appear as multiple events.
"""

import hashlib
from pathlib import Path

import torch


def deterministic_seed(*parts) -> int:
    key = "|".join(str(p) for p in parts)
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 1_000_000


# ============================================================
# events -> target matrices
# ============================================================

def events_to_targets(events, num_nodes):
    """
    routing[u, v]    = 1 if any event uses edge (u, v)  (kept for
                       diagnostics / backward compat; the informative
                       target is `load`, not this)
    load[u, v]       = COUNT of events using edge (u, v) -- how many
                       chunks crossed this edge. This is the target
                       that actually varies: across all three teachers,
                       routing_positive_ratio is ~0.99-1.00 (AllGather
                       on small/dense graphs uses nearly every edge at
                       least once), so "used or not" is close to
                       degenerate while per-edge load is not -- it is
                       also the quantity that determines makespan
                       (bottleneck edge = highest-load edge), so it is
                       the physically meaningful regression target.
    scheduling[u, v] = MIN epoch among events on (u, v), 0 if unused
    completion_epoch = max epoch over all events
    """
    routing = torch.zeros(num_nodes, num_nodes)
    load = torch.zeros(num_nodes, num_nodes)
    scheduling = torch.zeros(num_nodes, num_nodes)
    first = {}
    max_epoch = 0

    for ev in events:
        # Field names are NOT uniform across teachers: TE-CCL/SyCCL
        # events use src/dst for the physical edge; the ILP teacher's
        # events use src for the CHUNK ORIGIN (constant across a
        # chunk's whole multi-hop/multi-tree delivery) and from/to for
        # the actual per-hop edge, plus a fractional "amount" (ILP
        # splits a chunk across multiple parallel trees, e.g. 4 trees
        # x amount=0.25). from/to take priority when present so ILP's
        # "src" (origin, not edge) is never mistaken for an edge
        # endpoint; amount defaults to 1.0 (one whole chunk per event)
        # for teachers that don't report fractional sends.
        u = int(ev["from"]) if "from" in ev else int(ev["src"])
        v = int(ev["to"]) if "to" in ev else int(ev["dst"])
        ep = int(ev["epoch"])
        weight = float(ev.get("amount", 1.0))
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            raise ValueError(f"Event node out of range: {ev}")
        routing[u, v] = 1.0
        load[u, v] += weight
        key = (u, v)
        if key not in first or ep < first[key]:
            first[key] = ep
        max_epoch = max(max_epoch, ep)

    for (u, v), ep in first.items():
        scheduling[u, v] = float(ep)

    return routing, load, scheduling, int(max_epoch)


def derive_source_indices(events):
    """
    Node ids that ORIGINATE at least one chunk in this schedule, i.e.
    the functional "is this a real collective participant, or just a
    relay" signal — derived from ground truth, not from a topology's
    declared switch_indices (which may encode a physical switch, or a
    demand-model relay role assigned to an otherwise ordinary node;
    the two are not the same thing and must not be conflated into one
    feature bit downstream).

    Returns None when no event carries "chunk_origin" (adapters that
    don't populate it yet) — callers must fall back explicitly rather
    than silently treat "no data" as "no sources".
    """
    origins = {int(ev["chunk_origin"]) for ev in events
               if ev.get("chunk_origin") is not None}
    return sorted(origins) if origins else None


# ============================================================
# assembly + validation
# ============================================================

def build_sample(
    *, topology_name, collective, message_size_bytes,
    num_nodes, edge_index, capacity, latency, capacity_unit,
    events, switch_indices, source,
    solver_wall_time_sec=None, mip_gap=None, solver_status=None,
    completion_time=None, epoch_duration_sec=None, extra=None,
):
    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must be [2, E], "
                         f"got {tuple(edge_index.shape)}")
    E = edge_index.size(1)

    capacity = torch.as_tensor(capacity, dtype=torch.float).view(-1)
    latency = torch.as_tensor(latency, dtype=torch.float).view(-1)
    if capacity.numel() != E or latency.numel() != E:
        raise ValueError(
            f"capacity/latency length must equal E={E}, got "
            f"{capacity.numel()}/{latency.numel()}"
        )

    routing, load, scheduling, completion_epoch = events_to_targets(
        events, num_nodes
    )

    # every routed edge must exist in edge_index — a violated check
    # means the adapter's node-id mapping is wrong
    present = set(map(tuple, edge_index.t().tolist()))
    used = set(zip(*torch.nonzero(routing, as_tuple=True)))
    missing = {(int(u), int(v)) for u, v in used} - present
    if missing:
        raise ValueError(
            f"{topology_name}: schedule uses edges not in the "
            f"topology (adapter id-mapping bug?): "
            f"{sorted(missing)[:5]} ..."
        )

    sample = {
        "topology_name": str(topology_name),
        "collective": str(collective),
        "message_size": float(message_size_bytes),
        "node_feat": torch.zeros(num_nodes, 1),
        "edge_index": edge_index,
        "edge_attr": torch.stack([capacity, latency], dim=1),
        "capacity_unit": str(capacity_unit),
        "routing_target": routing,
        "load_target": load,
        "scheduling_target": scheduling,
        "completion_epoch": completion_epoch,
        "completion_time": completion_time,
        "schedule": list(events),
        "switch_indices": sorted(int(i) for i in switch_indices),
        "source_indices": derive_source_indices(events),
        "source": source,
        "solver_wall_time_sec": solver_wall_time_sec,
        "mip_gap": mip_gap,
        "solver_status": solver_status,
        "epoch_duration_sec": epoch_duration_sec,
    }
    if extra:
        sample.update(extra)
    return sample


# ============================================================
# incremental persistence (same pattern as the ILP generator)
# ============================================================

class IncrementalStore:
    def __init__(self, path: Path, key_fields):
        self.path = Path(path)
        self.key_fields = tuple(key_fields)
        if self.path.exists():
            self.samples = torch.load(self.path, weights_only=False)
        else:
            self.samples = []
        self.done = {self._key(s) for s in self.samples}
        print(f"[store] {self.path.name}: "
              f"{len(self.samples)} existing samples")

    def _key(self, s):
        return tuple(s.get(f) for f in self.key_fields)

    def has(self, sample_or_key):
        key = (self._key(sample_or_key)
               if isinstance(sample_or_key, dict) else
               tuple(sample_or_key))
        return key in self.done

    def add(self, sample):
        self.samples.append(sample)
        self.done.add(self._key(sample))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.samples, self.path)
