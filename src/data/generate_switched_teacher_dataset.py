"""
generate_switched_teacher_dataset.py — pilot sweep driver for the
parametric switched-topology ILP families (switched_topologies.py),
solved via the reconstructed solve_allgather_ilp (build_merged_dataset.py).

PILOT: 18 families x 1 message size (1MB) first -- observe per-family
solve time and failure rate before committing to the full 6-8-size
sweep (bigger families' MILP grows with node x tree x epoch, unlike
SyCCL's own combinatorial search, so wall-time is much less predictable
per family here).

Resilience: per-config wall-clock timeout + continue-on-failure, same
pattern as the SyCCL sweep (a single runaway family must not lose
progress on the rest).
"""

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.data.switched_topologies import family_grid          # noqa: E402
from src.data.build_merged_dataset import (                    # noqa: E402
    create_gurobi_env, solve_allgather_ilp,
)
from src.data.teacher_common import IncrementalStore            # noqa: E402

OUT_PATH = ROOT / "data" / "processed" / "switched_topo_teacher_dataset.pt"

PILOT_MESSAGE_SIZES = [2.5e5, 5e5, 1e6, 2e6, 4e6, 8e6, 16e6]
# same 7-point grid as generate_teacher_dataset.py's existing 5 ILP
# families (MESSAGE_SIZES) -- consistent convention across the whole
# ilp_teacher domain. Pilot (18 families x 1MB only) already validated:
# 18/18 solved, 0 failures, ~89 min total; full 18x7=126 grid estimated
# ~10h from that per-family average -- an overnight run.
SOLVE_TIME_LIMIT_SEC = 300      # per-config Gurobi TimeLimit
K = 60
TREE_NUMBER = 6
FRACTION_SPLIT = 3


def main():
    store = IncrementalStore(OUT_PATH, ("topology_name", "message_size"))
    env = create_gurobi_env(output_flag=0)

    grid = family_grid()
    # Reprioritize: the 3 TEST-holdout families (see
    # build_canonical_dataset.py's TEST_TOPOLOGIES) go first, so if
    # this overnight re-solve (fixing the chunk_size/num_sources bug)
    # doesn't finish all 18 families by morning, the ones that matter
    # for the reported CCT-gap table are done regardless.
    PRIORITY = {"dual_star_n12_r4", "two_tier_l4x4_r8", "swclust_2x8_r4"}
    grid = sorted(grid, key=lambda item: item[0] not in PRIORITY)
    print(f"{len(grid)} families x {len(PILOT_MESSAGE_SIZES)} sizes "
          f"= {len(grid) * len(PILOT_MESSAGE_SIZES)} candidate solves "
          f"(priority families first: {sorted(PRIORITY)})")

    n_ok = 0
    n_fail = 0
    for name, builder in grid:
        t = builder()
        for ms in PILOT_MESSAGE_SIZES:
            key = (f"switched_{name}", float(ms))
            if store.has(key):
                print(f"SKIP (already solved): {name} @ {ms:.0f}B")
                continue

            print(f"--- {name} (nodes={t.num_nodes}, "
                  f"gpus={len(t.gpu_indices)}, "
                  f"switches={len(t.switch_indices)}) @ {ms:.0f}B ---")
            t0 = time.time()
            try:
                sample = solve_allgather_ilp(
                    topo_demo=t.to_topo_demo(),
                    topology_name=f"switched_{name}",
                    message_size=ms,
                    seed=42,
                    gurobi_env=env,
                    capacity=t.capacity_bps,
                    transfer_delay=t.latency_s,
                    switch_indices=t.switch_indices,
                    source_indices=t.gpu_indices,
                    K=K,
                    tree_number=TREE_NUMBER,
                    fraction_split=FRACTION_SPLIT,
                    time_limit_sec=SOLVE_TIME_LIMIT_SEC,
                    verbose=False,
                )
                elapsed = time.time() - t0
                sample["variant_id"] = 0
                sample["link_variant_seed"] = 42
                sample["teacher_wall_time_sec"] = float(elapsed)
                store.add(sample)
                n_ok += 1
                print(f"  OK elapsed={elapsed:.1f}s status={sample['solver_status']} "
                      f"gap={sample['mip_gap']} completion_epoch={sample['completion_epoch']} "
                      f"saved={len(store.samples)}")
            except Exception as e:
                elapsed = time.time() - t0
                n_fail += 1
                print(f"  FAILED elapsed={elapsed:.1f}s error={e}")
                traceback.print_exc()

    print(f"\n=== PILOT DONE: {n_ok} ok, {n_fail} failed, "
          f"store has {len(store.samples)} total samples ===")


if __name__ == "__main__":
    main()
