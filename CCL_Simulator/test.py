import time
import simpy
import networkx as nx

from simcore import Sim, PolicyEntry


LINK_RATE = 100e9
PACKET_BYTES = 1500
HEADER_BYTES = 0

N_GPU = 16
CHUNK_MB = 64  # 压测调大；想快点结束调小


def build_ring_topology(n: int) -> nx.DiGraph:
    G = nx.DiGraph()

    for i in range(n):
        G.add_node(
            f"GPU{i}",
            type="gpu",
            num_qps=2,
            quantum_packets=1,
            tx_proc_delay=0.0,
            gpu_store_delay=0.0,
        )

    # one-direction ring: i -> (i+1)%n
    for i in range(n):
        for j in range(n):

            G.add_edge(f"GPU{i}", f"GPU{j}", link_rate_bps=LINK_RATE, prop_delay=0.0)

    return G


def make_ring_allgather_policy_no_deps(n: int, chunk_size_bytes: int, qpid: int = 0):
    """
    Ring allgather "traffic pattern" without dependencies:
      steps = n-1
      at time=0, each GPU injects (n-1) chunks to its next hop.
    This is purely for performance/stress testing of the simulator.
    """
    policy = []
    steps = n - 1

    for s in range(steps):
        for i in range(n):
            src = f"GPU{i}"
            dst = f"GPU{(i + 1) % n}"
            tx_name = f"AG_s{s}_i{i}"
            policy.append(
                PolicyEntry(
                    tx_name,
                    src,
                    dst,
                    qpid,
                    "Max",
                    chunk_size_bytes,
                    [src, dst],
                    time=0.0,
                )
            )
    return policy


def main():
    env = simpy.Environment()
    topo = build_ring_topology(N_GPU)

    sim = Sim(env, topo, packet_size_bytes=PACKET_BYTES, header_size_bytes=HEADER_BYTES)

    MB = 1024 * 1024
    chunk_size = CHUNK_MB * MB

    policy = make_ring_allgather_policy_no_deps(N_GPU, chunk_size_bytes=chunk_size, qpid=0)
    sim.load_policy(policy)

    sim.start()
    sim.run()

    makespan = max(sim.tx_complete_time.values())
    print(f"Total TXs: {len(sim.tx_complete_time)}")
    print(f"Makespan = {makespan:.6f} s")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    t1 = time.perf_counter()
    print(f"Total execution time: {t1 - t0:.3f} seconds")