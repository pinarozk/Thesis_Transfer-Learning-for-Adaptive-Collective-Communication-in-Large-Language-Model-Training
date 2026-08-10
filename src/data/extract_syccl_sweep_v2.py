"""
extract_syccl_sweep_v2.py — folds the message-size sweep results
(16-GPU + 32-GPU clos, AllGather, native SyCCL chunk sizes) into the
existing syccl_dataset.pt WITHOUT re-solving them via `synthesize`
(they're already solved -- re-running would just repeat the OOM wall
the 32-GPU family hit past 1MB). Reuses extract_syccl.py's topology
dump / route-table / event-parsing / build_sample machinery so there
is exactly one parser for the SyCCL result schema.

Usage: python extract_syccl_sweep_v2.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.data.extract_syccl import (              # noqa: E402
    get_topology_dump, build_topology_arrays, build_route_table,
    build_npu_to_device, parse_result_events, check_allgather_coverage,
    SYCCL_COLLECTIVE_NAMES, OUT_PATH,
)
from src.data.teacher_common import build_sample, IncrementalStore  # noqa: E402

RUNS_V2_DIR = ROOT / "data" / "raw" / "syccl_runs_v2"

FAMILIES = {
    "a1002": "clos16",   # 2 hosts x 8 gpu clos (paper's Figure 14 scale)
    "a100": "clos32",    # 4 hosts x 8 gpu clos (existing family, more sizes)
}


def run_one(cfg_path, result_path, family_tag, store):
    cfg = json.loads(cfg_path.read_text())
    size_tag = cfg_path.stem.replace("-config", "")
    topo_name = f"{family_tag}_{size_tag}"

    raw_coll = str(cfg.get("coll", {}).get("name", "allgather")).lower()
    coll = SYCCL_COLLECTIVE_NAMES.get(raw_coll, raw_coll)
    msg_bytes = float(cfg.get("coll", {}).get("byte", 0))

    key = (f"syccl_{topo_name}", msg_bytes)
    if store.has(key):
        print(f"SKIP (already in store): {topo_name}")
        return

    topo_dump = get_topology_dump(cfg, cfg_path)
    n, edge_index, cap, lat, switches = build_topology_arrays(topo_dump)
    route_table = build_route_table(topo_dump)
    npu_to_device = build_npu_to_device(topo_dump)
    gpu_ids = [d["id"] for d in topo_dump["devices"] if d["type"] == "GPU"]

    result = json.loads(result_path.read_text())
    events, completion_time_sec = parse_result_events(
        result, route_table, npu_to_device)

    if coll.lower() == "allgather":
        cov = check_allgather_coverage(events, gpu_ids)
        if not cov["ok"]:
            raise ValueError(
                f"AllGather coverage violated for {topo_name}: "
                f"absent origins={cov['origins_absent']}, "
                f"incomplete={ {k: v[:4] for k, v in list(cov['incomplete_origins'].items())[:3]} }"
            )

    sample = build_sample(
        topology_name=f"syccl_{topo_name}",
        collective=coll,
        message_size_bytes=msg_bytes,
        num_nodes=n,
        edge_index=edge_index,
        capacity=cap, latency=lat,
        capacity_unit="GBps_syccl",
        events=events,
        switch_indices=switches,
        source="syccl",
        solver_wall_time_sec=None,  # sweep script didn't record per-call wall time
        mip_gap=None,
        solver_status=None,
        completion_time=completion_time_sec,
        epoch_duration_sec=1e-9,
    )
    store.add(sample)
    print(f"OK {topo_name} | events={len(events)} "
          f"| t_complete={completion_time_sec:.6g}s "
          f"| saved={len(store.samples)}")


def main():
    store = IncrementalStore(OUT_PATH, ("topology_name", "message_size"))
    n_before = len(store.samples)

    n_pairs = 0
    n_errors = 0
    for family_dir, family_tag in FAMILIES.items():
        d = RUNS_V2_DIR / family_dir
        if not d.exists():
            print(f"SKIP missing dir: {d}")
            continue
        configs = sorted(d.glob("*-config.json"))
        for cfg_path in configs:
            result_path = d / cfg_path.name.replace("-config.json", "-result.json")
            if not result_path.exists():
                print(f"SKIP (no result, likely OOM/failed): {cfg_path.name}")
                continue
            n_pairs += 1
            try:
                run_one(cfg_path, result_path, family_tag, store)
            except Exception as e:
                n_errors += 1
                print(f"ERROR {cfg_path.name}: {e}")

    print(f"\n=== DONE: {n_pairs} pairs attempted, {n_errors} errors, "
          f"{len(store.samples) - n_before} new samples added "
          f"(store now has {len(store.samples)} total) ===")


if __name__ == "__main__":
    main()
