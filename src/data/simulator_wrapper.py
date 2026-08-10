import sys
from pathlib import Path

import simpy
import networkx as nx

# Add simulator folder to import path
ROOT_DIR = Path(__file__).resolve().parents[2]
SIMULATOR_DIR = ROOT_DIR / "CCL_Simulator"
sys.path.append(str(SIMULATOR_DIR))

from simcore import Sim, PolicyEntry


PACKET_BYTES = 1500
HEADER_BYTES = 0


def build_example_topology():
    G = nx.DiGraph()

    for i in range(4):
        G.add_node(
            f"GPU{i}",
            type="gpu",
            num_qps=2,
            quantum_packets=1,
            tx_proc_delay=0.0,
            gpu_store_delay=0.0,
        )

    link_rate = 100e9

    edges = [
        ("GPU0", "GPU1"),
        ("GPU1", "GPU0"),
        ("GPU1", "GPU2"),
        ("GPU2", "GPU1"),
        ("GPU1", "GPU3"),
        ("GPU3", "GPU1"),
    ]

    for src, dst in edges:
        G.add_edge(src, dst, link_rate_bps=link_rate, prop_delay=0.0)

    return G


def run_simulator(
    topology,
    policy_entries,
    packet_size_bytes=PACKET_BYTES,
    header_size_bytes=HEADER_BYTES,
    rate_updates=None,
):
    env = simpy.Environment()
    sim = Sim(
        env,
        topology,
        packet_size_bytes=packet_size_bytes,
        header_size_bytes=header_size_bytes,
    )

    sim.load_policy(policy_entries)

    if rate_updates is not None:
        sim.load_link_rate_schedule(rate_updates)

    sim.start()
    sim.run()

    if len(sim.tx_complete_time) == 0:
        raise RuntimeError("Simulator finished but no tx completion time was recorded.")

    makespan = max(sim.tx_complete_time.values())

    return makespan, dict(sim.tx_complete_time)


def build_example_policy(chunk_mb=64):
    MB = 1024 * 1024
    chunk_size = chunk_mb * MB

    policy = [
        PolicyEntry("A", "GPU0", "GPU1", 0, "Max", chunk_size, ["GPU0", "GPU1"], time=0.0),
        PolicyEntry("B", "GPU2", "GPU1", 0, "Max", chunk_size, ["GPU2", "GPU1"], time=1.0),
        PolicyEntry("C", "GPU1", "GPU3", 0, "Max", chunk_size, ["GPU1", "GPU3"], time=0.0, dependency=["A", "B"]),
        PolicyEntry("D", "GPU1", "GPU3", 0, "Max", chunk_size, ["GPU1", "GPU3"], time=0.0),
    ]

    return policy


if __name__ == "__main__":
    topo = build_example_topology()
    policy = build_example_policy(chunk_mb=64)

    rate_updates = {
        1.0: [("GPU1", "GPU3", 50e9)],
    }

    makespan, tx_times = run_simulator(topo, policy, rate_updates=rate_updates)

    print("Makespan:", makespan)
    print("TX times:", tx_times)