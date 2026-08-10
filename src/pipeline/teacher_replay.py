"""
teacher_replay.py — replay TEACHER schedules in the real simulator.

WHY (the measurement bug this fixes):
The CCT gap was defined as agent_sim_cct / teacher_reported_time. The
two sides of that ratio come from DIFFERENT time models: the teacher
number is the solver's own alpha-beta / epoch-model time (computed at
whatever capacity scale the solve used), the agent number is the event
simulator's time. Any unit or protocol mismatch between the two shows
up as a CONSTANT multiplicative offset per family -- exactly the
signature observed on the synthetic switched holdouts (gap ~ 99.4x
flat across a 64x message-size range; per-family constants 37/108/319
with ~2% std). Those numbers measured a unit inconsistency, not
schedule quality.

FIX: measure both sides in the SAME world. This module converts the
stored normalized teacher events into simulator PolicyEntries, so
cct_gap.py can compute

    gap_sim = agent_sim_cct / teacher_sim_cct

with identical topology conversion (to_simulator_topology) and an
identical time model on both sides. As a bonus this also removes the
long-standing TE-CCL epoch-model vs simulator protocol difference from
the DGX2 numbers. No retraining is needed -- this changes measurement
only.

Event normal form consumed (teacher_common):
    {"chunk": str, "chunk_origin": int|None, "src": int, "dst": int,
     "epoch": int, ["amount": float]}
'amount' (fractional multi-tree ILP events) scales the transfer size;
absent means 1.0.

Dependency semantics: a transfer of chunk c over (u, v) can start only
after c has ARRIVED at u -- i.e. after the latest earlier event of the
same chunk with dst == u. The chunk's origin needs no dependency at
its origin node. Epoch numbers are used only for ORDERING (they carry
the solver's discretization; wall-clock timing is the simulator's
job), plus a stable tie-break so replays are deterministic.
"""

import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "CCL_Simulator"))

from simcore import PolicyEntry                      # noqa: E402
from src.pipeline.schedule_bridge import (           # noqa: E402
    to_simulator_topology, run_ground_truth, _feat_idx_node,
)


def _n_sources(data):
    """Number of AllGather-participating nodes (is_source==1)."""
    is_source_i = _feat_idx_node(data, "is_source")
    is_source = data.x[:, is_source_i].numpy() > 0.5
    return max(int(is_source.sum()), 1)


def _per_source_chunk_bytes(data, message_size_bytes):
    """
    message_size_bytes convention is NOT uniform across teachers:
      - TE-CCL: message_size is the TOTAL AllGather buffer (sum
        across all sources); each source's own chunk -- what actually
        crosses one edge -- is message_size / n_sources. Verified
        numerically against TE-CCL's own epoch_duration_sec on
        DGX2_2_chassis (32 sources): epoch_duration * link_rate ==
        message_size_bytes / 32 exactly, on both a 16KB and a 1GB
        sample. Using the undivided total inflated every replayed
        transfer by n_sources (32x on DGX2) -- the missing factor
        behind the ~100x, message-size-growing replay_ratio there
        (bandwidth term dominates at large sizes, so the constant
        n_sources multiplier shows up growing toward it as latency's
        share shrinks).
      - ILP (original 5 families + switched_topologies.py) and SyCCL:
        message_size_bytes is ALREADY the per-source chunk (ILP's own
        generator scripts pass a per-node chunk_size directly; SyCCL's
        config_gen.py divides by nhosts*ngpus before writing
        chunk_size_B). Applying the /n_sources correction there too
        was tried and is WRONG -- confirmed by ring_cluster's
        replay_ratio dropping from a sane ~2.3x to ~0.24x (an exact
        8x = n_sources overcorrection) when the division was applied
        unconditionally.
    """
    if str(getattr(data, "source_domain", "")) == "teccl":
        return float(message_size_bytes) / _n_sources(data)
    return float(message_size_bytes)


def teacher_events_to_policy(events, message_size_bytes):
    """
    Normalized teacher events -> simulator PolicyEntry list.

    Returns (policy, stats). stats reports anything that had to be
    repaired so silent inconsistencies stay visible:
      * missing_arrival: transfers whose source node never received
        the chunk earlier in the schedule (and is not the origin) --
        scheduled without dependency, counted loudly;
      * fractional_events: events carrying amount != 1.
    """
    def _origin(e):
        # TE-CCL/SyCCL events carry an explicit chunk_origin. ILP
        # events don't have that key at all (chunk_number == 1 there,
        # so "chunk" itself is a constant '0' across every real
        # source) -- for ILP, "src" IS the origin (the field "from"/
        # "to" carry the real edge instead). Using chunk_origin alone
        # collapsed every ILP source into ONE group ('0', None),
        # merging unrelated chunks and reporting 100% missing_arrival
        # -- caught by the first real run on ring_cluster (371/371).
        co = e.get("chunk_origin")
        if co is not None:
            return int(co)
        if "from" in e:
            return int(e["src"])
        return None

    by_chunk = defaultdict(list)
    for e in events:
        key = (str(e.get("chunk", "?")), _origin(e))
        by_chunk[key].append(e)

    policy = []
    stats = {"missing_arrival": 0, "fractional_events": 0,
             "chunks": len(by_chunk), "entries": 0}
    eid = 0

    def _u(e):
        return int(e["from"]) if "from" in e else int(e["src"])

    def _v(e):
        return int(e["to"]) if "to" in e else int(e["dst"])

    for (chunk_id, origin), evs in by_chunk.items():
        # Field names are NOT uniform across teachers (see
        # teacher_common.events_to_targets): TE-CCL/SyCCL events use
        # src/dst for the physical edge; the ILP teacher's events
        # (original 5 families + switched_topologies.py) use src for
        # the CHUNK ORIGIN and from/to for the real per-hop edge.
        # from/to take priority when present so ILP's "src" is never
        # mistaken for an edge endpoint (caught by the first real run:
        # KeyError 'dst' on every ILP-domain replay).
        #
        # Ordering: TE-CCL's "epoch" does NOT strictly serialize
        # multi-hop chains -- a chunk's delivery to node X and X's
        # onward re-send can share the SAME epoch value (confirmed on
        # the DGX2_2_chassis holdout: epoch 0 contains both "15->0"
        # and "0->13" for the same chunk). A single (epoch, src, dst)
        # sort tie-breaks same-epoch hops by node id, not causal order,
        # so a re-send got scheduled before the delivery that enables
        # it -- this made ~100% of DGX2's events look like broken
        # dependencies (missing_arrival), silently making the replayed
        # teacher schedule start transfers before their chunk actually
        # arrived (artificially fast). Fixed-point scheduling instead:
        # repeatedly take whatever's resolvable (src already delivered)
        # in epoch order among the remainder, looping until a full pass
        # makes no progress. Only what's STILL unresolved after that is
        # a genuine inconsistency, not a tie-break artifact.
        remaining = sorted(evs, key=lambda e: (int(e["epoch"]), _u(e), _v(e)))
        delivered = {}
        if origin is not None:
            delivered[int(origin)] = None

        # Repeated-line fractional split (TE-CCL/SyCCL only -- events
        # with NO explicit "amount" key). TE-CCL's AllGather flow
        # format has no per-line volume field (unlike AlltoAll's "with
        # volume X"), yet the SAME (chunk, origin, u, v) edge can
        # appear many times across different epochs for one transfer
        # (confirmed on DGX2_2_chassis: edge 15->0 logged 15 times,
        # spread epoch 0-3) -- each line is a fractional flow-split
        # instance, not an independent full-size resend. Treating every
        # line as message_size_bytes inflated the replayed teacher CCT
        # by the repeat count (verified: teacher_sim_cct / teacher_
        # completion_time was ~100x on TE-CCL, up to 362x on some
        # graphs -- exactly this bug, not a real behavioral gap). ILP
        # events already carry a real per-epoch fractional "amount"
        # (its r[s,n,i,j,c,k] variable IS the epoch-k send fraction),
        # so they are read as-is, not further divided.
        edge_no_amount_counts = defaultdict(int)
        for e in evs:
            if "amount" not in e:
                edge_no_amount_counts[(_u(e), _v(e))] += 1

        def _emit(e, dep):
            nonlocal eid
            u, v = _u(e), _v(e)
            if "amount" in e:
                amount = float(e["amount"])
            else:
                amount = 1.0 / edge_no_amount_counts[(u, v)]
            if amount != 1.0:
                stats["fractional_events"] += 1
            entry_id = f"T_{chunk_id}_{origin}_{eid}"
            eid += 1
            policy.append(PolicyEntry(
                entry_id, f"GPU{u}", f"GPU{v}", qpid=0,
                rate="Max",
                chunk_size_bytes=message_size_bytes * amount,
                path=[f"GPU{u}", f"GPU{v}"], time=0.0,
                dependency=[dep] if dep else [],
            ))
            if v not in delivered:
                delivered[v] = entry_id

        while remaining:
            progressed = False
            still_remaining = []
            for e in remaining:
                u = _u(e)
                if u in delivered:
                    _emit(e, delivered[u])
                    progressed = True
                else:
                    still_remaining.append(e)
            remaining = still_remaining
            if not progressed:
                break

        # whatever's left after convergence is a genuine, unresolved
        # inconsistency (not a tie-break artifact) -- schedule without
        # a dependency but count it loudly.
        for e in remaining:
            stats["missing_arrival"] += 1
            _emit(e, None)
        stats["entries"] += len(evs)

    return policy, stats


def teacher_sim_cct(data, sample_schedule, message_size_bytes):
    """
    One-call: canonical graph + stored teacher events -> simulated
    teacher completion time (seconds), using the SAME topology
    conversion as the agent side.

    message_size_bytes is expected exactly as stored on the sample
    (data.message_size_bytes) -- this function applies the correct,
    teacher-specific per-source-chunk conversion itself (see
    _per_source_chunk_bytes), so callers should NOT pre-divide.
    """
    topo = to_simulator_topology(data)
    per_source_chunk = _per_source_chunk_bytes(data, message_size_bytes)
    policy, stats = teacher_events_to_policy(sample_schedule,
                                             per_source_chunk)
    cct = run_ground_truth(topo, policy)
    return cct, stats
