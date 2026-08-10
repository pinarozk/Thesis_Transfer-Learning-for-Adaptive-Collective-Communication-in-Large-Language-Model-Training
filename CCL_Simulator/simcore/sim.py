from __future__ import annotations

import simpy
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union, Iterable

from .nodes import NodeConfig, GPUNode, SwitchNode, BaseNode
from .types import PolicyEntry, TxId, Packet
from .policy import PolicyEngine, PolicySpec


# schedule: time -> [(u, v, new_rate_bps), ...]
LinkRateSchedule = Dict[float, List[Tuple[str, str, float]]]


class Sim:
    """Policy-driven packet-level simulator."""

    def __init__(
        self,
        env: simpy.Environment,
        topo: nx.DiGraph,
        packet_size_bytes: int = 1500,
        header_size_bytes: int = 0,
        link_rate_schedule: Optional[LinkRateSchedule] = None,
    ):
        self.env = env
        self.topo = topo

        self.nodes: Dict[str, BaseNode] = {}

        self.tx_complete_time: Dict[TxId, float] = {}
        self.chunk_ready_time: Dict[Tuple[Union[int, str], str], float] = {}

        self.policy = PolicyEngine(
            env,
            self,
            PolicySpec(packet_size_bytes=packet_size_bytes, header_size_bytes=header_size_bytes),
        )

        # Optional runtime link-rate updates. If None or {}, link rates stay constant.
        self._link_rate_schedule: Optional[LinkRateSchedule] = link_rate_schedule

        self._build_nodes_and_ports()
        self.tx_first_send_time: Dict[TxId, float] = {}


    def load_link_rate_schedule(self, schedule: Optional[LinkRateSchedule]) -> None:
        """Optional. Provide a dict: time -> list[(u, v, rate_bps)].
        If schedule is None or empty, link rates remain constant.
        """
        self._link_rate_schedule = schedule

    def _build_nodes_and_ports(self) -> None:
        # Nodes
        for nid, attrs in self.topo.nodes(data=True):
            ntype = attrs.get("type", None)
            if ntype not in ("gpu", "switch"):
                raise ValueError(f"Node {nid} must have attr type='gpu' or 'switch'")

            cfg = NodeConfig(
                node_id=nid,
                node_type=ntype,
                num_qps=int(attrs.get("num_qps", 1)),
                quantum_packets=int(attrs.get("quantum_packets", 1)),
                tx_proc_delay=float(attrs.get("tx_proc_delay", 0.0)),
                sw_proc_delay=float(attrs.get("sw_proc_delay", 0.0)),
                gpu_store_delay=float(attrs.get("gpu_store_delay", 0.0)),
            )

            if ntype == "switch":
                self.nodes[nid] = SwitchNode(self.env, cfg)
            else:
                self.nodes[nid] = GPUNode(
                    self.env,
                    cfg,
                    on_tx_complete=self._on_tx_complete,
                    on_chunk_ready=self._on_chunk_ready,
                )

        # Ports per edge
        for u, v, eattr in self.topo.edges(data=True):
            rate = float(eattr.get("link_rate_bps", 0.0))
            delay = float(eattr.get("prop_delay", 0.0))
            if rate <= 0:
                raise ValueError(f"Edge {u}->{v} needs link_rate_bps > 0")
            if delay < 0:
                raise ValueError(f"Edge {u}->{v} needs prop_delay >= 0")

            src = self.nodes[u]

            def deliver_fn(pkt: Packet, dst_id=v):
                self.nodes[dst_id].receive(pkt)

            if src.cfg.node_type == "switch":
                num_qps = 1
                quantum = 1
            else:
                num_qps = src.cfg.num_qps
                quantum = src.cfg.quantum_packets

            src.add_port(
                next_hop_id=v,
                link_rate_bps=rate,
                prop_delay=delay,
                deliver_fn=deliver_fn,
                num_qps=num_qps,
                quantum_packets=quantum,
                tx_proc_delay=src.cfg.tx_proc_delay,
                header_size_bytes=self.policy.spec.header_size_bytes,
            )

    # ---- Policy API ----
    def load_policy(self, entries: Iterable[PolicyEntry]) -> None:
        self.policy.install(entries)

    def start(self) -> None:
        # Start link-rate update process (optional)
        if self._link_rate_schedule:
            self.env.process(self._run_link_rate_updates(self._link_rate_schedule))
        self.policy.bootstrap()

    # ---- Link-rate updates ----
    def _set_link_rate(self, u: str, v: str, rate_bps: float) -> None:
        if u not in self.nodes:
            raise KeyError(f"Unknown node {u}")
        if v not in self.nodes:
            raise KeyError(f"Unknown node {v}")

        src = self.nodes[u]
        port = src.ports.get(v, None)
        if port is None:
            raise KeyError(f"No directed link/port {u}->{v}")

        # Update the Port's rate
        port.set_link_rate_bps(rate_bps)

        # Keep topology edge attr in sync (useful for debug/export)
        if self.topo.has_edge(u, v):
            self.topo[u][v]["link_rate_bps"] = float(rate_bps)

    def _run_link_rate_updates(self, schedule: LinkRateSchedule):
        # Apply updates in chronological order
        for t in sorted(schedule.keys()):
            t = float(t)
            if t > self.env.now:
                yield self.env.timeout(t - self.env.now)
            # t <= now: apply immediately
            for (u, v, rate) in schedule[t]:
                self._set_link_rate(u, v, rate)

    # ---- Injection ----
    def register_tx(self, tx_id: TxId) -> None:
        if tx_id not in self.tx_complete_time:
            self.tx_complete_time[tx_id] = float("nan")

    def send_from_src(self, pkt: Packet) -> None:
        src_id = pkt.path[pkt.hop_idx]
        node = self.nodes[src_id]
        if node.cfg.node_type != "gpu":
            raise ValueError("Policy src must be a GPU")
        if pkt.seq == 0 and pkt.tx_id not in self.tx_first_send_time:
            self.tx_first_send_time[pkt.tx_id] = self.env.now
        node._send_to_next(pkt)

    # ---- Callbacks ----
    def _on_tx_complete(self, tx_id: TxId, t: float) -> None:
        # store once
        old = self.tx_complete_time.get(tx_id, float("nan"))
        if old != old:  # NaN check
            self.tx_complete_time[tx_id] = t

    def _on_chunk_ready(self, node_id: str, chunk_id: Union[int, str], t: float) -> None:
        key = (chunk_id, node_id)
        if key not in self.chunk_ready_time:
            self.chunk_ready_time[key] = t
        self.policy.on_chunk_ready(node_id, chunk_id)

    def run(self, until: Optional[float] = None) -> None:
        self.env.run(until=until)