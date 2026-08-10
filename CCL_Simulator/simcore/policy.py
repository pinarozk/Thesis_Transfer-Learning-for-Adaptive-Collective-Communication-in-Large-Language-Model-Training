from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Iterable, Set

import simpy

from .types import PolicyEntry, Packet, TxId


@dataclass(frozen=True)
class PolicySpec:
    packet_size_bytes: int = 1500
    header_size_bytes: int = 0


class PolicyEngine:
    """Policy-driven injection.

    Installs rules keyed by (chunk_id, src).
    When (src, chunk_id) becomes ready, schedule all rules at that src.
    Each rule fires when BOTH conditions hold:
      - env.now >= e.time (release time)
      - all chunks in e.dependency are ready at e.src
    """

    def __init__(self, env: simpy.Environment, sim: "Sim", spec: PolicySpec):
        self.env = env
        self.sim = sim
        self.spec = spec

        self.rules: Dict[Tuple[Union[int, str], str], List[PolicyEntry]] = {}
        self._fired: Set[Tuple[Union[int, str], str]] = set()  # (chunk_id, src)
        self._scheduled_entries: Set[int] = set()

        # NEW: per-(node,chunk) readiness events for dependency gating
        self._ready_events: Dict[Tuple[str, Union[int, str]], simpy.Event] = {}
        self._ready_marked: Set[Tuple[str, Union[int, str]]] = set()

    def install(self, entries: Iterable[PolicyEntry]) -> None:
        for e in entries:
            e.validate()
            key = (e.chunk_id, e.src)
            self.rules.setdefault(key, []).append(e)

    def infer_initial_sources(self) -> Dict[Union[int, str], List[str]]:
        by_chunk_src: Dict[Union[int, str], set[str]] = {}
        by_chunk_dst: Dict[Union[int, str], set[str]] = {}

        for (chunk_id, src), lst in self.rules.items():
            by_chunk_src.setdefault(chunk_id, set()).add(src)
            for e in lst:
                by_chunk_dst.setdefault(chunk_id, set()).add(e.dst)

        initial: Dict[Union[int, str], List[str]] = {}
        for chunk_id, srcs in by_chunk_src.items():
            dsts = by_chunk_dst.get(chunk_id, set())
            init = sorted(list(srcs - dsts))
            if not init:
                init = sorted(list(srcs))
            initial[chunk_id] = init
        return initial

    def bootstrap(self) -> None:
        initial = self.infer_initial_sources()
        for chunk_id, srcs in initial.items():
            for s in srcs:
                node = self.sim.nodes[s]
                if node.cfg.node_type != "gpu":
                    raise ValueError(f"Initial source {s} for chunk {chunk_id} must be a GPU")
                node.mark_initial_chunk(chunk_id)
                # Mark ready + schedule any outgoing entries from (s, chunk_id)
                self.on_chunk_ready(s, chunk_id)

    # ---- Dependency readiness bookkeeping ----
    def _ready_event(self, node_id: str, chunk_id: Union[int, str]) -> simpy.Event:
        key = (node_id, chunk_id)
        ev = self._ready_events.get(key)
        if ev is None:
            ev = simpy.Event(self.env)
            self._ready_events[key] = ev
        return ev

    def _mark_ready(self, node_id: str, chunk_id: Union[int, str]) -> None:
        key = (node_id, chunk_id)
        if key in self._ready_marked:
            return
        self._ready_marked.add(key)

        ev = self._ready_events.get(key)
        if ev is None:
            ev = simpy.Event(self.env)
            self._ready_events[key] = ev
        if not ev.triggered:
            ev.succeed()

    # ---- Runtime hook from simulator ----
    def on_chunk_ready(self, node_id: str, chunk_id: Union[int, str]) -> None:
        # NEW: mark this (node, chunk) ready for dependency gating
        self._mark_ready(node_id, chunk_id)

        key = (chunk_id, node_id)
        if key in self._fired:
            return
        self._fired.add(key)

        for e in self.rules.get(key, []):
            eid = id(e)
            if eid in self._scheduled_entries:
                continue
            self._scheduled_entries.add(eid)
            self.env.process(self._fire_entry_when_allowed(e))

    def _fire_entry_when_allowed(self, e: PolicyEntry):
        # 1) wait until earliest time
        wait = max(0.0, float(e.time) - self.env.now)
        if wait > 0:
            yield self.env.timeout(wait)

        # 2) wait until all dependencies are ready at e.src
        deps = e.dependency or []
        if deps:
            evs = [self._ready_event(e.src, dep_chunk) for dep_chunk in deps]
            yield simpy.events.AllOf(self.env, evs)

        # 3) fire
        self._fire_entry(e)

    def _fire_entry(self, e: PolicyEntry) -> None:
        rate_bps, use_max_rate = e.normalized_rate()

        ps = int(self.spec.packet_size_bytes)
        # chunk_size_bytes is a float in practice (message sizes come
        # from canonical/PyG data as float64); "//" on a float
        # operand returns a float, and range() below requires int --
        # pre-existing bug, surfaced by the first real
        # agent-decoded-schedule run through this simulator.
        total_packets = int((e.chunk_size_bytes + ps - 1) // ps)
        total_packets = max(1, total_packets)

        tx_id: TxId = (e.chunk_id, e.src, e.dst)
        self.sim.register_tx(tx_id)

        for i in range(total_packets):
            remaining = e.chunk_size_bytes - i * ps
            sz = ps if remaining >= ps else remaining
            if sz <= 0:
                sz = ps

            pkt = Packet(
                tx_id=tx_id,
                chunk_id=e.chunk_id,
                tx_src=e.src,
                tx_dst=e.dst,
                seq=i,
                total_packets=total_packets,
                size_bytes=sz,
                path=list(e.path),
                hop_idx=0,
                qpid=int(e.qpid),
                rate_bps=float(rate_bps),
                use_max_rate=bool(use_max_rate),
                created_time=self.env.now,
            )
            self.sim.send_from_src(pkt)