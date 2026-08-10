"""
extract_syccl_expand2.py — folds the second SyCCL expansion round
(clos16 AlltoAll full native range 4KB-8GB, clos32 AllGather 4MB) into
syccl_dataset.pt. Reuses extract_syccl_sweep_v2.py's run_one()
unchanged. IncrementalStore's own dedup (topology_name, message_size)
skips the 5 clos16 a2a sizes already folded in earlier.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.data.extract_syccl import OUT_PATH               # noqa: E402
from src.data.extract_syccl_sweep_v2 import run_one         # noqa: E402
from src.data.teacher_common import IncrementalStore         # noqa: E402

SOURCES = [
    (ROOT / "data" / "raw" / "syccl_runs_a2a_v2", "clos16_a2a"),
    (ROOT / "data" / "raw" / "syccl_runs_clos32_ag_v2", "clos32"),
]


def main():
    store = IncrementalStore(OUT_PATH, ("topology_name", "message_size"))
    n_before = len(store.samples)

    n_errors = 0
    for runs_dir, family_tag in SOURCES:
        configs = sorted(runs_dir.glob("*-config.json"))
        for cfg_path in configs:
            result_path = runs_dir / cfg_path.name.replace("-config.json", "-result.json")
            if not result_path.exists():
                print(f"SKIP (no result): {cfg_path.name}")
                continue
            try:
                run_one(cfg_path, result_path, family_tag, store)
            except Exception as e:
                n_errors += 1
                print(f"ERROR {cfg_path.name}: {e}")

    print(f"\n=== DONE: {n_errors} errors, "
          f"{len(store.samples) - n_before} new samples added "
          f"(store now has {len(store.samples)} total) ===")


if __name__ == "__main__":
    main()
