"""
Incremental AllGather ILP teacher dataset generator.

Fixes vs. previous version
--------------------------
1. DETERMINISTIC SEEDS: Python's built-in hash() is salted per process
   (PYTHONHASHSEED), so the old seeds were NOT reproducible across runs.
   We now derive seeds from hashlib.md5, which is stable forever.
2. TEACHER QUALITY METADATA: we record wall-clock solver time for every
   sample and carry through 'mip_gap' / 'solver_status' if the solver
   provides them. These are needed later to (a) filter or down-weight
   low-quality labels, (b) report the ILP-vs-Agent speedup honestly.
3. MORE LINK VARIANTS: configurable, default raised from 3 to 10 to give
   group-based splits enough parameter diversity per topology family.

NOTE for solve_allgather_ilp:
    If possible, populate sample["mip_gap"] (model.MIPGap) and
    sample["solver_status"] (model.Status) inside the solver function.
    This script will store them if present and fall back gracefully
    if not.
"""

import time
import hashlib
from pathlib import Path

import torch

from data.build_merged_dataset import (
    BASE_DIR,
    create_gurobi_env,
    build_original_8gpu_topology,
    build_original_plus_extra_topology,
    build_two_bridge_topology,
    build_ring_cluster_topology,
    build_soft_leaf_spine_topology,
    solve_allgather_ilp,
)

# ============================================================
# Config
# ============================================================

NUM_LINK_VARIANTS = 10          # was 3 — more parameter diversity
ILP_TIME_LIMIT_SEC = 300
MESSAGE_SIZES = [2.5e5, 5e5, 1e6, 2e6, 4e6, 8e6, 16e6]

TOPOLOGY_BUILDERS = [
    ("original_8gpu", build_original_8gpu_topology),
    ("original_plus_extra", build_original_plus_extra_topology),
    ("two_bridge", build_two_bridge_topology),
    ("ring_cluster", build_ring_cluster_topology),
    ("soft_leaf_spine", build_soft_leaf_spine_topology),
]


# ============================================================
# Deterministic seed (reproducible across runs / machines)
# ============================================================

def deterministic_seed(*parts) -> int:
    key = "|".join(str(p) for p in parts)
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 1_000_000


# ============================================================
# Main
# ============================================================

def main():
    output_dir = BASE_DIR / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / "allgather_teacher_dataset_incremental.pt"
    failed_path = output_dir / "allgather_teacher_failed_incremental.pt"

    if dataset_path.exists():
        samples = torch.load(dataset_path, weights_only=False)
        print(f"Loaded existing dataset: {len(samples)} samples")
    else:
        samples = []

    if failed_path.exists():
        failed = torch.load(failed_path, weights_only=False)
        print(f"Loaded existing failed cases: {len(failed)}")
    else:
        failed = []

    completed_keys = set()
    for sample in samples:
        completed_keys.add(
            (
                sample["topology_name"],
                int(sample.get("variant_id", 0)),
                float(sample["message_size"]),
            )
        )

    env = create_gurobi_env()

    print("=" * 70)
    print("Generating incremental AllGather ILP teacher dataset")
    print("Deterministic seeds: hashlib-based (reproducible).")
    print("Topology families:", [name for name, _ in TOPOLOGY_BUILDERS])
    print("Message sizes:", MESSAGE_SIZES)
    print("Link variants:", NUM_LINK_VARIANTS)
    print("Saving to:", dataset_path)
    print("=" * 70)

    for topo_name, topo_fn in TOPOLOGY_BUILDERS:
        topo = topo_fn()

        for variant_id in range(NUM_LINK_VARIANTS):
            for ms in MESSAGE_SIZES:
                sample_key = (topo_name, int(variant_id), float(ms))

                if sample_key in completed_keys:
                    print("SKIP", "| topo:", topo_name,
                          "| variant:", variant_id, "| size:", ms)
                    continue

                # Reproducible: same (topo, variant, size) -> same seed,
                # in every run, on every machine.
                seed = deterministic_seed(topo_name, variant_id, ms)

                try:
                    start = time.time()

                    sample = solve_allgather_ilp(
                        topo_demo=topo,
                        topology_name=topo_name,
                        message_size=ms,
                        seed=seed,
                        gurobi_env=env,
                        K=80,
                        tree_number=8,
                        fraction_split=4,
                        time_limit_sec=ILP_TIME_LIMIT_SEC,
                        verbose=False,
                    )

                    elapsed = time.time() - start

                    # ---- provenance & teacher-quality metadata ----
                    sample["variant_id"] = int(variant_id)
                    sample["link_variant_seed"] = int(seed)
                    sample["teacher_wall_time_sec"] = float(elapsed)
                    sample["teacher_time_limit_sec"] = float(ILP_TIME_LIMIT_SEC)
                    # If the solver populated these, keep them; otherwise
                    # mark explicitly as unknown (None), never guess.
                    sample.setdefault("mip_gap", None)
                    sample.setdefault("solver_status", None)

                    samples.append(sample)
                    completed_keys.add(sample_key)

                    torch.save(samples, dataset_path)
                    torch.save(failed, failed_path)

                    print(
                        "OK",
                        "| topo:", topo_name,
                        "| variant:", variant_id,
                        "| size:", ms,
                        "| epoch:", sample["completion_epoch"],
                        "| events:", len(sample["schedule"]),
                        "| gap:", sample.get("mip_gap"),
                        "| elapsed:", round(elapsed, 2),
                        "| saved:", len(samples),
                    )

                except Exception as e:
                    failed.append({
                        "topology_name": topo_name,
                        "variant_id": int(variant_id),
                        "message_size": float(ms),
                        "seed": int(seed),
                        "error": str(e),
                    })

                    torch.save(samples, dataset_path)
                    torch.save(failed, failed_path)

                    print("FAILED", "| topo:", topo_name,
                          "| variant:", variant_id,
                          "| size:", ms, "| error:", e)

    torch.save(samples, dataset_path)
    torch.save(failed, failed_path)

    print("=" * 70)
    print("DONE")
    print("Generated samples:", len(samples))
    print("Failed:", len(failed))
    print("Saved:", dataset_path)
    print("Saved:", failed_path)


if __name__ == "__main__":
    main()
