from __future__ import annotations
import simpy
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Union

from .port import Port, LinkSpec
from .types import Packet, TxId


@dataclass(slots=True)
class NodeConfig:
    node_id: str
    node_type: str  # "gpu" or "switch"
    num_qps: int = 1
    quantum_packets: int = 1
    tx_proc_delay: float = 0.0
    sw_proc_delay: float = 0.0
    gpu_store_delay: float = 0.0


class BaseNode:
    def __init__(self, env: simpy.Environment, cfg: NodeConfig):
        self.env = env
        self.cfg = cfg
        self.node_id = cfg.node_id
        self.ports: Dict[str, Port] = {}  # next_hop_id -> Port

    def add_port(
        self,
        next_hop_id: str,
        link_rate_bps: float,
        prop_delay: float,
        deliver_fn: Callable[[Packet], None],
        num_qps: int,
        quantum_packets: int,
        tx_proc_delay: float,
        header_size_bytes:int,
    ) -> None:
        self.ports[next_hop_id] = Port(
            env=self.env,
            owner_id=self.node_id,
            next_hop_id=next_hop_id,
            link=LinkSpec(link_rate_bps=link_rate_bps, prop_delay=prop_delay),
            deliver_fn=deliver_fn,
            num_qps=num_qps,
            quantum_packets=quantum_packets,
            tx_proc_delay=tx_proc_delay,
            header_size_bytes=header_size_bytes
        )

    def _send_to_next(self, pkt: Packet) -> None:
        nxt = pkt.next_hop()
        if nxt is None:
            return
        outp = self.ports.get(nxt)
        if outp is None:
            raise KeyError(f"{self.node_id} has no port to {nxt}")
        pkt.advance()
        outp.enqueue(pkt, qpid=pkt.qpid)

    def receive(self, pkt: Packet) -> None:
        raise NotImplementedError


class SwitchNode(BaseNode):
    def receive(self, pkt: Packet) -> None:
        self.env.process(self._handle(pkt))

    def _handle(self, pkt: Packet):
        if self.cfg.sw_proc_delay > 0:
            yield self.env.timeout(self.cfg.sw_proc_delay)
        self._send_to_next(pkt)


class GPUNode(BaseNode):
    """GPU behavior:

    - If packet.tx_dst == this GPU: count packets for pkt.tx_id.
      When all packets arrive, tx completes; chunk becomes available (ready) at this GPU.
    - Otherwise: relay GPU does packet-level forwarding immediately.
    """

    def __init__(
        self,
        env: simpy.Environment,
        cfg: NodeConfig,
        on_tx_complete: Callable[[TxId, float], None],
        on_chunk_ready: Callable[[str, Union[int, str], float], None],
    ):
        super().__init__(env, cfg)
        self.on_tx_complete = on_tx_complete
        self.on_chunk_ready = on_chunk_ready

        self.have_chunk: Dict[Union[int, str], bool] = {}
        self._rx_cnt: Dict[TxId, int] = {}

    def mark_initial_chunk(self, chunk_id: Union[int, str]) -> None:
        self.have_chunk[chunk_id] = True

    def receive(self, pkt: Packet) -> None:
        self.env.process(self._handle(pkt))

    def _handle(self, pkt: Packet):
        if pkt.tx_dst == self.node_id:
            tx = pkt.tx_id
            self._rx_cnt[tx] = self._rx_cnt.get(tx, 0) + 1
            if self._rx_cnt[tx] >= pkt.total_packets:
                if self.cfg.gpu_store_delay > 0:
                    yield self.env.timeout(self.cfg.gpu_store_delay)

                self.on_tx_complete(tx, self.env.now)

                if not self.have_chunk.get(pkt.chunk_id, False):
                    self.have_chunk[pkt.chunk_id] = True
                    self.on_chunk_ready(self.node_id, pkt.chunk_id, self.env.now)
            return

        # relay
        self._send_to_next(pkt)
