from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

RateInput = Union[float, int, str]
ChunkId = Union[int, str]


@dataclass(frozen=True)
class PolicyEntry:
    """One policy rule:
      [chunk_id, src, dst, qpid, rate, chunk_size_bytes, path_nodes, time, dependency]
    """
    chunk_id: ChunkId
    src: str
    dst: str
    qpid: int
    rate: RateInput        # number (bps) or "Max"
    chunk_size_bytes: int
    path: List[str]
    time: float = 0.0  # earliest trigger time (seconds)

    # NEW: dependency chunks that must be ready at src before this entry can fire
    dependency: List[ChunkId] = field(default_factory=list)

    def normalized_rate(self) -> tuple[float, bool]:
        if isinstance(self.rate, str):
            if self.rate.strip().lower() == "max":
                return 0.0, True
            raise ValueError(f"Invalid rate string: {self.rate!r}. Use number (bps) or 'Max'.")
        r = float(self.rate)
        if r <= 0:
            raise ValueError("rate must be > 0, or 'Max'")
        return r, False

    def validate(self) -> None:
        if not self.path or self.path[0] != self.src or self.path[-1] != self.dst:
            raise ValueError(f"Path must start at src and end at dst for entry chunk={self.chunk_id}: {self.path}")
        if self.chunk_size_bytes <= 0:
            raise ValueError("chunk_size_bytes must be > 0")
        if self.qpid < 0:
            raise ValueError("qpid must be >= 0")

        # dependency is optional; if provided, must be a list
        if self.dependency is None:
            raise ValueError("dependency must be omitted or a list (cannot be None)")
        if not isinstance(self.dependency, list):
            raise ValueError("dependency must be a list")
        # (optional sanity) avoid self-dependency which can deadlock if used naively
        if self.chunk_id in self.dependency:
            raise ValueError(f"dependency must not contain itself (chunk_id={self.chunk_id})")


TxId = Tuple[ChunkId, str, str]  # (chunk_id, src, dst)


@dataclass(slots=True)
class Packet:
    # Transmission identity
    tx_id: TxId
    chunk_id: ChunkId
    tx_src: str
    tx_dst: str

    # Packet sequencing
    seq: int
    total_packets: int

    # Size
    size_bytes: int

    # Routing
    path: List[str]
    hop_idx: int  # index of current node in path

    # Scheduling
    qpid: int
    rate_bps: float
    use_max_rate: bool

    created_time: float = 0.0

    def next_hop(self) -> Optional[str]:
        if self.hop_idx + 1 >= len(self.path):
            return None
        return self.path[self.hop_idx + 1]

    def advance(self) -> None:
        self.hop_idx += 1

    @property
    def bits(self) -> int:
        return self.size_bytes * 8