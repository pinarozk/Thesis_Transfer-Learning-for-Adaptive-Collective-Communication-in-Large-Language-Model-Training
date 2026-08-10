"""
extract_syccl_a2a.py — folds the AlltoAll-on-Clos SyCCL sweep
(16-GPU clos, 5 sizes 4KB-1MB) into syccl_dataset.pt. Reuses
extract_syccl_sweep_v2.py's run_one() unchanged (already branches
check_allgather_coverage only for coll=="allgather").
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.data.extract_syccl import OUT_PATH               # noqa: E402
from src.data.extract_syccl_sweep_v2 import run_one        # noqa: E402
from src.data.teacher_common import IncrementalStore        # noqa: E402

RUNS_DIR = ROOT / "data" / "raw" / "syccl_runs_a2a"


def main():
    store = IncrementalStore(OUT_PATH, ("topology_name", "message_size"))
    n_before = len(store.samples)

    configs = sorted(RUNS_DIR.glob("*-config.json"))
    n_errors = 0
    for cfg_path in configs:
        result_path = RUNS_DIR / cfg_path.name.replace("-config.json", "-result.json")
        if not result_path.exists():
            print(f"SKIP (no result): {cfg_path.name}")
            continue
        try:
            run_one(cfg_path, result_path, "clos16_a2a", store)
        except Exception as e:
            n_errors += 1
            print(f"ERROR {cfg_path.name}: {e}")

    print(f"\n=== DONE: {n_errors} errors, "
          f"{len(store.samples) - n_before} new samples added "
          f"(store now has {len(store.samples)} total) ===")


if __name__ == "__main__":
    main()
