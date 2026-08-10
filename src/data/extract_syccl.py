"""
extract_syccl.py — aliyun/SyCCL teacher adapter.
(repo: https://github.com/aliyun/syccl, SIGCOMM 2025;
 symmetry-based decomposition, SCIP-backed MILP/LP)

FORMAT NOTE (verified against a REAL `synthesize solve` run, not
guessed — cross-checked against both the writer (src/Sketch/Output.cpp)
and an independent reader (tests.hpp's reSim) in SyCCL's own source):

    result_json = {
      "coll_name": "allgather", "ngpus": N, "chunk_size_byte": B,
      "alg_times": [...], "num_algs": K,
      "algorithms": [
        { "final_schedule": {
            "Time": <float, MICROSECONDS>,
            "Schedule": {
              "init_devs": [[[chunk_origin, subidx], [dev_ids...]], ...],
              "chunk_sizes": [[[chunk_origin, subidx], size_bytes], ...],
              "Events": [
                {"src_chunk": "(origin, subidx)",
                 "sends": [{"src_gpu":i,"dst_gpu":j,"epoch":e,
                            "copy":bool,"reduce":bool}, ...]},
                ...
              ]
            }
        }},
        ...
      ]  # sorted ascending by final_schedule.Time -> [0] is BEST
    }

TWO THINGS THAT ARE NOT WHAT THEY LOOK LIKE (both verified against
real solver output, both would silently corrupt labels if assumed
naively):

1. "epoch" in each send is NOT a small ordinal index like TE-CCL's —
   it is already a raw TIME value in nanoseconds (confirmed against
   dump_topo's latency_ns convention and the solver's own "Epoch
   duration: X ns" log lines). epoch_duration_sec is therefore fixed
   at 1e-9, not derived per-sample.

2. "src_gpu"/"dst_gpu" in a send are LOGICAL endpoints, not
   necessarily a direct physical edge — SyCCL's topology has NIC and
   switch devices in between (confirmed: GPU 0's only real topology
   edges go to its local NIC and local NVSwitch, never to another GPU
   directly). A cross-host send like "GPU0 -> GPU13" is a multi-hop
   physical path (GPU0 -> NIC -> net-switch -> NIC -> GPU13) collapsed
   into one logical log line — the TE-CCL "via switches" problem's
   SyCCL cousin. We resolve the REAL hop chain via the topology's own
   Topology::getNpuRoute() (exposed through dump_topo's "gpu_routes"
   table), not by reimplementing their routing logic.

REQUIRES A WSL BUILD (see thesis notes): SyCCL is C++/SCIP-based, no
Windows build path. The `synthesize` and `dump_topo` binaries are
built inside WSL (Ubuntu-24.04, /root/SyCCL/build/); this script
shells out to them via `wsl.exe`. dump_topo is a small helper added
alongside SyCCL (not upstream) that dumps Topology::devices/links and
all-pairs Topology::getNpuRoute() results as JSON — it exists
specifically so we consume SyCCL's OWN verified topology/routing
code instead of re-deriving the layered hosts/topo config format
ourselves (that format's hyperedge/switch-level semantics are load-
bearing and easy to get subtly wrong).

Usage:
    python extract_syccl.py --inspect-result <result.json>
    python extract_syccl.py --dry-parse <result.json> --topo <topo.json>
    python extract_syccl.py                              # full sweep
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.data.teacher_common import (            # noqa: E402
    build_sample, IncrementalStore,
)

# ============================================================
# Config — WSL-side paths (binaries only exist inside WSL)
# ============================================================

WSL_DISTRO = "Ubuntu-24.04"
SYCCL_BUILD_DIR_WSL = "/root/SyCCL/build"
SYNTHESIZE_BIN_WSL = f"{SYCCL_BUILD_DIR_WSL}/synthesize"
DUMP_TOPO_BIN_WSL = f"{SYCCL_BUILD_DIR_WSL}/dump_topo"

CONFIG_DIR = ROOT / "teacher_inputs" / "syccl"     # config JSONs (Windows side)
WORK_DIR = ROOT / "data" / "raw" / "syccl_runs"
TOPO_CACHE_DIR = ROOT / "data" / "raw" / "syccl_topo_cache"
OUT_PATH = ROOT / "data" / "processed" / "syccl_dataset.pt"

TIMEOUT_SEC = 3600


def win_to_wsl_path(p: Path) -> str:
    """C:\\Users\\X\\... -> /mnt/c/Users/X/... (this repo lives on C:)."""
    s = str(p.resolve())
    drive, rest = s.split(":", 1)
    return f"/mnt/{drive.lower()}" + rest.replace("\\", "/")


def wait_for_file(path: Path, tries=20, delay=0.25) -> bool:
    """
    WSL2's DrvFs (the /mnt/c bridge) can lag slightly before a file a
    WSL process just closed becomes visible to Windows-side Path.
    exists() -- fast solves (~1-2s) hit this race in practice (slower
    ones happened to have enough incidental buffer not to). Poll
    briefly instead of failing on a false negative.
    """
    for _ in range(tries):
        if path.exists():
            return True
        time.sleep(delay)
    return path.exists()


def run_wsl(args, timeout=TIMEOUT_SEC):
    """Run a command inside WSL; returns CompletedProcess."""
    cmd = ["wsl.exe", "-d", WSL_DISTRO, "-u", "root", "--"] + args
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout)


# ============================================================
# Topology: obtained from SyCCL's OWN code via dump_topo, never
# hand-parsed from the layered hosts/topo config format.
# ============================================================

def topo_signature(cfg):
    """
    Topology depends only on hosts/topo/host_links, not on message
    size or collective — same signature -> reuse the dump (mirrors
    TE-CCL's chassis-level sidecar reuse). Hashed, not the raw JSON:
    the topo layer list contains ':' and '"' which are illegal in
    Windows filenames (WSL's own write silently "succeeds" against
    the mangled name while Windows-side Path.exists() then reports
    the file missing -- a real failure mode hit once already).
    """
    h = cfg.get("hosts", {})
    key = (h.get("host_num"), h.get("host_gpu_num"), h.get("host_nic_num"),
          h.get("host_links"), json.dumps(cfg.get("topo", []), sort_keys=True))
    raw = "_".join(str(x) for x in key)
    human = f"{h.get('host_num')}h_{h.get('host_gpu_num')}g_{h.get('host_links')}"
    digest = hashlib.md5(raw.encode()).hexdigest()[:10]
    return f"{human}_{digest}"


def get_topology_dump(cfg, cfg_path):
    """Returns the parsed dump_topo JSON, caching per topo_signature."""
    sig = topo_signature(cfg)
    cache_path = TOPO_CACHE_DIR / f"{sig}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    TOPO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_wsl = win_to_wsl_path(cache_path)
    cfg_wsl = win_to_wsl_path(cfg_path)
    proc = run_wsl([DUMP_TOPO_BIN_WSL, cfg_wsl, out_wsl])
    if proc.returncode != 0 or not wait_for_file(cache_path):
        raise RuntimeError(
            f"dump_topo failed for {cfg_path.name}: "
            f"{proc.stderr[-800:] if proc.stderr else 'no output'}"
        )
    return json.loads(cache_path.read_text())


def build_topology_arrays(topo_dump):
    """
    dump_topo's device/edge dump -> (num_nodes, edge_index, capacity,
    latency, switch_indices). Self-loops (the "memcpy" local layer,
    src==dst) are excluded -- they are not real inter-node links.
    NIC/SWITCH_* devices are non-source relays (switch_indices); GPU
    devices are the AllGather participants.
    """
    n = topo_dump["num_devices"]
    switch_indices = sorted(
        d["id"] for d in topo_dump["devices"] if d["type"] != "GPU"
    )
    src, dst, cap, lat = [], [], [], []
    for e in topo_dump["edges"]:
        if e["src"] == e["dst"]:
            continue
        src.append(e["src"]); dst.append(e["dst"])
        cap.append(float(e["bandwidth_GBps"]))
        lat.append(float(e["latency_ns"]) * 1e-9)   # ns -> s
    return n, [src, dst], cap, lat, switch_indices


def build_route_table(topo_dump):
    """(src_gpu, dst_gpu) -> [device_id, ...] real physical hop chain.
    Keys are NPU-logical ids (0..num_npus-1), matching Events'
    src_gpu/dst_gpu — see npu_to_device for why that is NOT the same
    as Device::device_id."""
    return {
        (r["src_gpu"], r["dst_gpu"]): r["path"]
        for r in topo_dump["gpu_routes"]
    }


def build_npu_to_device(topo_dump):
    """NPU-logical id (0..num_npus-1) -> real (sparse) Device::device_id.
    Read directly from Topology::npus by dump_topo — NOT the same
    mapping once NICs/switches are interposed between GPUs of
    different hosts (verified: host1's GPUs have npu_id 8..15 but
    device_id 13..20 in the default 2-host nvswitch config)."""
    return {i: dev_id for i, dev_id in
            enumerate(topo_dump["npu_to_device"])}


# ============================================================
# Result parsing (verified schema — see module docstring)
# ============================================================

_SRC_CHUNK_RE = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")


def parse_src_chunk(s):
    m = _SRC_CHUNK_RE.match(s)
    if not m:
        raise ValueError(f"Unrecognized src_chunk format: {s!r}")
    return int(m.group(1)), int(m.group(2))


def pick_best_algorithm(result):
    """
    algorithms[] is sorted ascending by final_schedule.Time by SyCCL
    itself (Algorithm.cpp sorts generated_algos before emitting), so
    index 0 is the best — but we don't trust that silently, we verify.
    """
    algos = result.get("algorithms")
    if not algos:
        raise ValueError(f"No 'algorithms' in result. Keys: "
                         f"{sorted(result.keys())}")
    times = [a["final_schedule"]["Time"] for a in algos]
    best_idx = min(range(len(times)), key=lambda i: times[i])
    if best_idx != 0:
        print(f"  NOTE: algorithms[] not pre-sorted in this result "
              f"(best is index {best_idx}, time={times[best_idx]}); "
              f"using it anyway.")
    return algos[best_idx]


def parse_result_events(result, route_table, npu_to_device=None):
    """
    Best algorithm's Schedule.Events -> normalized events, with every
    logical (src_gpu, dst_gpu) send EXPANDED into its real physical
    hop chain via route_table (see get_topology_dump / getNpuRoute).
    chunk_origin is translated NPU-logical -> real device id (via
    npu_to_device) so it lives in the same id space as src/dst -- the
    canonical builder's is_source feature depends on that consistency.
    Returns (events, completion_time_sec).
    """
    algo = pick_best_algorithm(result)
    fs = algo["final_schedule"]
    sched = fs["Schedule"]
    completion_time_sec = float(fs["Time"]) * 1e-6   # us -> s

    events = []
    unmapped_routes = 0
    for grp in sched["Events"]:
        origin_npu, subidx = parse_src_chunk(grp["src_chunk"])
        origin = (npu_to_device.get(origin_npu, origin_npu)
                 if npu_to_device else origin_npu)
        for send in grp["sends"]:
            if send.get("reduce"):
                # AllGather never reduces; a reduce=True send would
                # mean this isn't a pure copy-forward we can model as
                # a routing edge the way we do -- surface it loudly
                # rather than silently mis-recording it as a copy.
                raise ValueError(
                    f"Unexpected reduce=True send in {grp['src_chunk']}; "
                    f"this adapter only handles copy-forward events."
                )
            s, d = int(send["src_gpu"]), int(send["dst_gpu"])
            epoch = float(send["epoch"])   # already ns, see module note
            path = route_table.get((s, d))
            if path is None:
                unmapped_routes += 1
                path = [s, d]   # fall back to logical edge (will be
                                # caught by build_sample's edge-in-
                                # topology check if it's not real)
            for i in range(len(path) - 1):
                events.append({
                    "chunk": f"{origin}.{subidx}",
                    "chunk_origin": origin,
                    "src": path[i],
                    "dst": path[i + 1],
                    "epoch": epoch,
                })
    if unmapped_routes:
        print(f"  WARNING: {unmapped_routes} sends had no route in "
              f"gpu_routes (used direct logical edge as fallback).")
    return events, completion_time_sec


# ============================================================
# AllGather coverage sanity (same invariant as extract_teccl.py)
# ============================================================

def check_allgather_coverage(events, gpu_ids):
    participants = set(gpu_ids)
    per_origin = {}
    for e in events:
        per_origin.setdefault(e["chunk_origin"], set()).add(
            (e["src"], e["dst"]))
    origins_seen = set(per_origin)
    incomplete = {}
    for o, edges in per_origin.items():
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
        reached, stack = {o}, [o]
        while stack:
            u = stack.pop()
            for v in adj.get(u, ()):
                if v not in reached:
                    reached.add(v)
                    stack.append(v)
        missing = participants - reached
        if missing:
            incomplete[o] = sorted(missing)
    return {
        "origins_absent": sorted(participants - origins_seen),
        "incomplete_origins": incomplete,
        "ok": not incomplete and not (participants - origins_seen),
    }


# ============================================================
# Driver
# ============================================================

SYCCL_COLLECTIVE_NAMES = {
    "allgather": "AllGather", "alltoall": "AlltoAll",
    "allreduce": "AllReduce", "reducescatter": "ReduceScatter",
    "broadcast": "Broadcast", "gather": "Gather", "scatter": "Scatter",
    "sendrecv": "SendRecv",
}


def run_one(cfg_path, store):
    cfg = json.loads(cfg_path.read_text())
    topo_name = cfg_path.stem
    # normalize casing to match the other teachers' convention
    # ("AllGather", not SyCCL's own lowercase "allgather") -- the
    # canonical builder's is_allgather/is_alltoall flags compare
    # against exact capitalized strings.
    raw_coll = str(cfg.get("coll", {}).get("name", "allgather")).lower()
    coll = SYCCL_COLLECTIVE_NAMES.get(raw_coll, raw_coll)
    msg_bytes = float(cfg.get("coll", {}).get("byte", 0))

    key = (f"syccl_{topo_name}", msg_bytes)
    if store.has(key):
        print(f"SKIP {topo_name}")
        return

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    out_json = WORK_DIR / f"{topo_name}_result.json"
    run_cfg = dict(cfg)
    # NOTE: solve_output lives inside "algo_solve", not top-level (a
    # top-level key of the same name is silently ignored by SyCCL --
    # confirmed against a real config; setting only the top-level key
    # made synthesize write to the config's own unmodified relative
    # "./result-*.json", resolved against WSL's default cwd, not ours).
    run_cfg.setdefault("algo_solve", {})["solve_output"] = \
        win_to_wsl_path(out_json)
    run_cfg_path = WORK_DIR / f"{topo_name}_config.json"
    run_cfg_path.write_text(json.dumps(run_cfg, indent=2))

    t0 = time.time()
    proc = run_wsl([SYNTHESIZE_BIN_WSL, "-f", win_to_wsl_path(run_cfg_path),
                    "solve"])
    elapsed = time.time() - t0
    if proc.returncode != 0 or not wait_for_file(out_json):
        raise RuntimeError(
            f"synthesize failed for {topo_name}: returncode="
            f"{proc.returncode} stdout_tail={proc.stdout[-400:]!r} "
            f"stderr_tail={proc.stderr[-400:]!r}"
        )

    topo_dump = get_topology_dump(cfg, cfg_path)
    n, edge_index, cap, lat, switches = build_topology_arrays(topo_dump)
    route_table = build_route_table(topo_dump)
    npu_to_device = build_npu_to_device(topo_dump)
    gpu_ids = [d["id"] for d in topo_dump["devices"] if d["type"] == "GPU"]

    result = json.loads(out_json.read_text())
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
        solver_wall_time_sec=elapsed,
        mip_gap=None,
        solver_status=None,
        completion_time=completion_time_sec,
        epoch_duration_sec=1e-9,   # events already carry raw-ns epochs
    )
    store.add(sample)
    print(f"OK {topo_name} | events={len(events)} "
          f"| t_complete={completion_time_sec:.6g}s | {elapsed:.1f}s "
          f"| saved={len(store.samples)}")


# ============================================================
# Inspect / dry-parse
# ============================================================

def inspect_result(path):
    obj = json.loads(Path(path).read_text())

    def walk(o, prefix="", depth=0):
        if depth > 3:
            return
        if isinstance(o, dict):
            for k in list(o.keys())[:15]:
                print(f"{prefix}{k}  ->  {type(o[k]).__name__}")
                walk(o[k], prefix + "    ", depth + 1)
        elif isinstance(o, list) and o:
            print(f"{prefix}[list of {len(o)}] first: "
                  f"{str(o[0])[:150]}")
            if isinstance(o[0], dict):
                walk(o[0], prefix + "    ", depth + 1)

    walk(obj)


def dry_parse(result_path, topo_path):
    result = json.loads(Path(result_path).read_text())
    topo_dump = json.loads(Path(topo_path).read_text())
    route_table = build_route_table(topo_dump)
    npu_to_device = build_npu_to_device(topo_dump)
    gpu_ids = [d["id"] for d in topo_dump["devices"] if d["type"] == "GPU"]

    events, completion_time_sec = parse_result_events(
        result, route_table, npu_to_device)
    n, edge_index, cap, lat, switches = build_topology_arrays(topo_dump)
    present = set(zip(edge_index[0], edge_index[1]))
    missing = {(e["src"], e["dst"]) for e in events} - present

    print(f"parsed events (post route-expansion): {len(events)}")
    print(f"completion_time_sec: {completion_time_sec:.6g}")
    print(f"edges used but not in topology: {sorted(missing)[:10]} "
          f"({len(missing)} total)")
    cov = check_allgather_coverage(events, gpu_ids)
    print(f"AllGather coverage: {'OK' if cov['ok'] else 'VIOLATED'}")
    print(f"  origins absent: {cov['origins_absent']}")
    if cov["incomplete_origins"]:
        print(f"  incomplete: "
              f"{ {k: v for k, v in list(cov['incomplete_origins'].items())[:5]} }")
    print("\nfirst 5 events:")
    for e in events[:5]:
        print(" ", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect-result", type=str, default=None)
    ap.add_argument("--dry-parse", type=str, default=None,
                    help="result.json (requires --topo)")
    ap.add_argument("--topo", type=str, default=None,
                    help="topo.json produced by dump_topo")
    args = ap.parse_args()

    if args.inspect_result:
        inspect_result(args.inspect_result)
        return
    if args.dry_parse:
        if not args.topo:
            raise SystemExit("--dry-parse requires --topo <topo.json>")
        dry_parse(args.dry_parse, args.topo)
        return

    configs = sorted(CONFIG_DIR.glob("*.json"))
    if not configs:
        raise FileNotFoundError(
            f"No SyCCL configs in {CONFIG_DIR} — copy some from "
            f"external/SyCCL/config/ first."
        )
    store = IncrementalStore(OUT_PATH, ("topology_name", "message_size"))
    for cfg_path in configs:
        try:
            run_one(cfg_path, store)
        except Exception as e:
            print(f"FAILED {cfg_path.stem}: {e}")

    print(f"\nDONE — {len(store.samples)} samples in {OUT_PATH}")


if __name__ == "__main__":
    main()
