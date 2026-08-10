import time
import simpy
import networkx as nx

from simcore import Sim, PolicyEntry


LINK_RATE_0 = 100e9
LINK_RATE_1 = 50e9

PACKET_BYTES = 1500
HEADER_BYTES = 0

CHUNK_MB = 64  # 你也可以改小一点让测试更快，比如 4


def build_topology():
    G = nx.DiGraph()

    # only 2 GPUs, direct link both directions (optional)
    G.add_node("GPU0", type="gpu", num_qps=2, quantum_packets=1, tx_proc_delay=0.0, gpu_store_delay=0.0)
    G.add_node("GPU1", type="gpu", num_qps=2, quantum_packets=1, tx_proc_delay=0.0, gpu_store_delay=0.0)
    G.add_node("GPU2", type="gpu", num_qps=2, quantum_packets=1, tx_proc_delay=0.0, gpu_store_delay=0.0)
    G.add_node("GPU3", type="gpu", num_qps=2, quantum_packets=1, tx_proc_delay=0.0, gpu_store_delay=0.0)


    # directed edges
    G.add_edge("GPU0", "GPU1", link_rate_bps=LINK_RATE_0, prop_delay=0.0)
    G.add_edge("GPU1", "GPU0", link_rate_bps=LINK_RATE_0, prop_delay=0.0)
    G.add_edge("GPU1", "GPU2", link_rate_bps=LINK_RATE_0, prop_delay=0.0)
    G.add_edge("GPU2", "GPU1", link_rate_bps=LINK_RATE_0, prop_delay=0.0)
    G.add_edge("GPU1", "GPU3", link_rate_bps=LINK_RATE_0, prop_delay=0.0)
    G.add_edge("GPU3", "GPU1", link_rate_bps=LINK_RATE_0, prop_delay=0.0)

    return G


def main():
    env = simpy.Environment()
    topo = build_topology()

    sim = Sim(env, topo, packet_size_bytes=PACKET_BYTES, header_size_bytes=HEADER_BYTES)

    MB = 1024 * 1024
    chunk_size = CHUNK_MB * MB

    # Two flows on the same link GPU0->GPU1:
    # - flow A at t=0.0s
    # - flow B at t=1.0s
    policy = [
        # [chunkid, src, dst, qpid, rate, chunksize, path, time]
        PolicyEntry("A", "GPU0", "GPU1", 0, "Max", chunk_size, ["GPU0", "GPU1"], time=0.0),
        PolicyEntry("B", "GPU2", "GPU1", 0, "Max", chunk_size, ["GPU2", "GPU1"], time=1.0),
        PolicyEntry("C", "GPU1", "GPU3", 0, "Max", chunk_size, ["GPU1", "GPU3"], time=0.0, dependency=["A", "B"]),
        PolicyEntry("D", "GPU1", "GPU3", 0, "Max", chunk_size, ["GPU1", "GPU3"], time=0.0),
        # PolicyEntry("E", "GPU0", "GPU1", 0, "Max", chunk_size, ["GPU0", "GPU1"], time=2.0),
        # PolicyEntry("E", "GPU1", "GPU2", 0, "Max", chunk_size, ["GPU1", "GPU2"], time=2.0)
    ]
    sim.load_policy(policy)

    # At t=1.0s, reduce GPU0->GPU1 rate
    rate_updates = {
        1.0: [("GPU1", "GPU3", LINK_RATE_1)],
    }
    sim.load_link_rate_schedule(rate_updates)

    sim.start()
    sim.run()

    print("=== TX completion times ===")
    for tx_id, t in sorted(sim.tx_complete_time.items(), key=lambda x: x[1]):
        print(f"tx={tx_id} complete: {t:.6f} s")

    makespan = max(sim.tx_complete_time.values())
    print(f"\nMakespan = {makespan:.6f} s")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    t1 = time.perf_counter()
    print(f"\nTotal execution time: {t1 - t0:.3f} seconds")