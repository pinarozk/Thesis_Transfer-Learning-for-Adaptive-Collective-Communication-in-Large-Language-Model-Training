from __future__ import annotations

import math
import simpy
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Optional

from .types import Packet


@dataclass(frozen=True)
class LinkSpec:
    link_rate_bps: float
    prop_delay: float


@dataclass
class Port:
    """One directed output link == one Port.

    - Single-server: sends one packet at a time.
    - Multiple QPs: RR across non-empty QPs, with configurable quantum (#packets per visit).
    - Per-packet service time: (pkt.size_bytes + header) * 8 / eff_rate,
      where eff_rate = link_rate if use_max_rate else min(pkt.rate_bps, link_rate).
    - Prop delay: after service, packet arrives at next hop after prop_delay.

    Optimization changes vs original:
      1) No always-alive _run() process. We start a drain process only when queue transitions empty->non-empty.
      2) No per-packet env.process(_deliver_after). We schedule delivery via timeout callbacks.
      3) Track total queued packets (_nq) to avoid scanning qps in _has_data().
    """

    def __init__(
        self,
        env: simpy.Environment,
        owner_id: str,
        next_hop_id: str,
        link: LinkSpec,
        deliver_fn: Callable[[Packet], None],
        num_qps: int = 1,
        quantum_packets: int = 1,
        tx_proc_delay: float = 0.0,
        header_size_bytes: int = 0,
    ):
        self.env = env
        self.owner_id = owner_id
        self.next_hop_id = next_hop_id
        self.link = link
        self.deliver_fn = deliver_fn

        self.num_qps = max(1, int(num_qps))
        self.quantum_packets = max(1, int(quantum_packets))
        self.tx_proc_delay = max(0.0, float(tx_proc_delay))

        self.qps: Dict[int, Deque[Packet]] = {i: deque() for i in range(self.num_qps)}
        self._rr = 0
        self.header_size_bytes = max(0, int(header_size_bytes))

        # New: on-demand drain process + total queued counter
        self._proc = None  # simpy.events.Process | None
        self._nq = 0       # total number of queued packets across all QPs

    def set_link_rate_bps(self, new_rate_bps: float) -> None:
        """Update this directed link's line rate at runtime.

        Semantics: affects packets whose service starts after this update.
        Packets already in service keep their previously computed service time.
        """
        r = float(new_rate_bps)
        if not math.isfinite(r) or r <= 0:
            raise ValueError(
                f"new link_rate_bps must be finite and > 0 on {self.owner_id}->{self.next_hop_id}, got {new_rate_bps}"
            )

        # LinkSpec is frozen=True; replace the whole object.
        self.link = LinkSpec(link_rate_bps=r, prop_delay=self.link.prop_delay)

    def enqueue(self, pkt: Packet, qpid: int) -> None:
        q = self.qps[int(qpid) % self.num_qps]
        was_empty = (self._nq == 0)

        q.append(pkt)
        self._nq += 1

        # Start a drain process only on empty->non-empty transition.
        if was_empty and self._proc is None:
            self._proc = self.env.process(self._drain())

    def _has_data(self) -> bool:
        return self._nq > 0

    def _next_non_empty_qp(self) -> Optional[int]:
        # RR scan: O(num_qps). If num_qps is large, we can optimize further later.
        for i in range(self.num_qps):
            idx = (self._rr + i) % self.num_qps
            if self.qps[idx]:
                return idx
        return None

    def _service_time(self, pkt: Packet) -> float:
        link_rate = self.link.link_rate_bps
        if link_rate <= 0:
            raise ValueError(f"link_rate_bps must be > 0 on {self.owner_id}->{self.next_hop_id}")

        if pkt.use_max_rate:
            eff = link_rate
        else:
            if pkt.rate_bps <= 0:
                raise ValueError(f"rate_bps must be > 0 for tx_id={pkt.tx_id}")
            eff = min(pkt.rate_bps, link_rate)

        total_bits = (pkt.size_bytes + self.header_size_bytes) * 8
        return total_bits / eff

    def _schedule_deliver(self, pkt: Packet, delay: float) -> None:
        if delay <= 0:
            self.deliver_fn(pkt)
            return

        evt = self.env.timeout(delay)
        # callback signature: cb(event). Bind pkt via default arg.
        evt.callbacks.append(lambda _ev, p=pkt: self.deliver_fn(p))

    def _drain(self):
        # Drain until empty, then exit (no permanent generator).
        while self._nq > 0:
            qp = self._next_non_empty_qp()
            if qp is None:
                break

            sent = 0
            while sent < self.quantum_packets and self.qps[qp]:
                pkt = self.qps[qp].popleft()
                self._nq -= 1

                if self.tx_proc_delay > 0:
                    yield self.env.timeout(self.tx_proc_delay)

                st = self._service_time(pkt)
                if st > 0:
                    yield self.env.timeout(st)

                pd = self.link.prop_delay
                if pd < 0:
                    raise ValueError("prop_delay must be >= 0")

                self._schedule_deliver(pkt, pd)
                sent += 1

            self._rr = (qp + 1) % self.num_qps

        # Mark idle so next enqueue can start a new drain process.
        self._proc = None