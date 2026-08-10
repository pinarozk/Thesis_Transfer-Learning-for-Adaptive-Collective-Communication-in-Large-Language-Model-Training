"""
extract_teccl.py — microsoft/TE-CCL teacher adapter.
(repo: https://github.com/microsoft/TE-CCL, SIGCOMM 2024)

FORMAT NOTE (verified against a real ndv2_schedule.json, not guessed):
TE-CCL's schedule JSON uses NUMBER-PREFIXED top-level keys and stores
flows as human-readable STRINGS, not dicts:

    {
      "1-Epoch_Duration": <float seconds>,
      ...
      "7-Flows": [
         "Chunk 0 from 13 traveled over 13->12 in epoch 0",
         ...
      ],
      "8-Chunk paths": [ "10->8 in epoch 0", ... ]
    }

So the primary parser is regex-based over the "*-Flows" list (route A).
Dict-shaped layouts from other versions/forks are still handled
(routes B/C), so this adapter survives format drift.

BONUS from the real format: "1-Epoch_Duration" gives the epoch length
in SECONDS. It is recorded as epoch_duration_sec and used to derive
completion_time, which makes epochs and completion times COMPARABLE
ACROSS TEACHERS (TE-CCL epochs and our own ILP epochs are different
discretizations of the same physical time). This closes the
"are the epoch definitions comparable?" gap flagged earlier.

Usage:
    python extract_teccl.py --inspect   <schedule.json>  # structure
    python extract_teccl.py --dry-parse <schedule.json>  # parse only
    python extract_teccl.py                              # full sweep

REMAINING ADAPTER POINT: parse_topology_from_input(). Route (a) reads
inline capacity/alpha matrices from TopologyParams; route (b) reads a
small sidecar you write once per topology. For NDv2/DGX2 the sidecar
is the fastest, zero-ambiguity path and it also retires the hardcoded
switch indices in the canonical builder.
"""

import argparse
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
# Config
# ============================================================

TEMPLATE_DIR = ROOT / "teacher_inputs" / "teccl"
WORK_DIR = ROOT / "data" / "raw" / "teccl_runs"
OUT_PATH = ROOT / "data" / "processed" / "TE_CCL_dataset.pt"

CHUNK_SIZES_BYTES = [2.5e5, 1e6, 4e6, 16e6]

# ============================================================
# Pre-solved TE-CCL paper outputs (experiments/output_provided/)
# ============================================================
# These are already-solved schedules shipped with the TE-CCL repo
# (SIGCOMM'24 paper experiments) — no Gurobi re-solve needed, only
# parsing. Folder layout (see teccl/examples/README.md):
#   {DGX2,NDv2}_output/{2_chassis,4_chassis}/{AllGather,AlltoAll}/
#       {Fast,Fast_Early_Stop,Slow}/{size}.json
# "Fast"/"Slow" = epoch_type (FASTEST_LINK/SLOWEST_LINK) — a solver
# discretization choice, NOT a different physical topology.
# "Fast_Early_Stop" = SAME topology+demand as "Fast", solved with a
# looser mip_gap=0.3 (see json_gen.py) — a genuine lower-quality
# label for the same problem, not a duplicate to deduplicate away.
PROVIDED_DIR = (ROOT / "external" / "TE-CCL" / "teccl" / "examples"
                / "experiments" / "output_provided")

# (topology, chassis_folder) -> input template whose sidecar
# ('<template>.topology.json') has the physical-unit topology.
PROVIDED_SIDECAR_MAP = {
    ("DGX2", "2_chassis"): TEMPLATE_DIR / "dgx2_input.json",
    ("NDv2", "2_chassis"): TEMPLATE_DIR / "ndv2_input.json",
    ("NDv2", "4_chassis"): TEMPLATE_DIR / "ndv2_chassis4_input.json",
}

# json_gen.py: mip_gap=0.3 only for Fast_Early_Stop, else the sample
# templates' default GurobiParams.mip_gap (1e-4).
PROVIDED_MIP_GAP = {"Fast_Early_Stop": 0.3}
PROVIDED_MIP_GAP_DEFAULT = 1e-4

# json_gen.py: tts (hence chunk/message size) is in GB, DECIMAL
# (1e-6 GB == 1KB == 1000 bytes) — NOT the 1024-based convention used
# elsewhere (e.g. build_canonical_dataset.safe_float_message_size).
# Using the wrong base silently shifts every provided-output message
# size by up to ~7% per unit step — small, but exactly the kind of
# cross-source scale drift this whole adapter effort exists to avoid.
_DECIMAL_SIZE_RE = re.compile(r"^(?P<num>[\d.]+)(?P<unit>[KMG]?B)$")
_DECIMAL_SIZE_MULT = {"B": 1, "KB": 1_000, "MB": 1_000_000,
                      "GB": 1_000_000_000}


def parse_provided_size_bytes(name: str) -> float:
    m = _DECIMAL_SIZE_RE.match(name)
    if not m:
        raise ValueError(f"Unrecognized provided-output size token: "
                         f"{name!r} (expected e.g. '16KB', '1GB')")
    return float(m.group("num")) * _DECIMAL_SIZE_MULT[m.group("unit")]
TECCL_CMD = ["teccl", "solve", "--input_args"]
TIMEOUT_SEC = 3600


# ============================================================
# Numbered-key helpers ("7-Flows" -> "flows")
# ============================================================

_NUM_PREFIX = re.compile(r"^\s*\d+\s*[-_.:]\s*")


def norm_key(k) -> str:
    return _NUM_PREFIX.sub("", str(k)).strip().lower().replace(" ", "_")


def key_index(d: dict) -> dict:
    """normalized name -> original key"""
    return {norm_key(k): k for k in d}


def find_key(d: dict, *candidates):
    idx = key_index(d)
    for c in candidates:
        if c in idx:
            return idx[c]
    for c in candidates:                     # substring fallback
        for nk, orig in idx.items():
            if c in nk:
                return orig
    return None


def to_float(v):
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(v))
    return float(m.group()) if m else None


# ============================================================
# Route A — string flow log (the real TE-CCL format)
# ============================================================

# NOTE: TE-CCL's own chunk_flow_path_to_string() (allgather.py) MERGES
# multi-hop paths that pass through a switch into ONE logged line and
# appends "via switches A -> B -> ...": e.g.
#   "Chunk 0 from 1 traveled over 1->4 in epoch 0 via switches 0"
# really means the PHYSICAL path 1->0->4, not a direct 1->4 link. If
# this suffix is dropped, the parser hands build_sample() an edge that
# does not exist in the topology (build_sample's own edge-in-topology
# check WILL catch this — verified against a real solve: 5/58 edges
# in the ndv2 sample were switch-compressed and failed exactly this
# check before this fix). We must expand the compressed hop back into
# the real per-hop chain: src -> switch_1 -> ... -> dst.
FLOW_RE = re.compile(
    r"chunk\s+(?P<chunk>\S+)\s+from\s+(?P<origin>\d+)\s+"
    r"traveled\s+over\s+(?P<src>\d+)\s*->\s*(?P<dst>\d+)\s+"
    # AlltoAll flow lines insert "with volume X" before "in epoch"
    # (AllGather lines don't); volume is the fraction of the chunk
    # carried by this hop, not currently consumed downstream but kept
    # so we don't silently drop it.
    r"(?:with\s+volume\s+(?P<volume>[\d.]+)\s+)?"
    r"in\s+epoch\s+(?P<epoch>\d+)"
    r"(?:\s+via\s+switches\s+(?P<switches>[\d\s>-]+))?",
    re.IGNORECASE,
)

LOOSE_RE = re.compile(                       # "a->b ... epoch k"
    r"(?P<src>\d+)\s*->\s*(?P<dst>\d+).*?epoch\s+(?P<epoch>\d+)",
    re.IGNORECASE,
)


def _expand_switch_hops(origin, src, dst, epoch, chunk, switches_str):
    """
    "1->4 ... via switches 0" -> physical hops [1->0, 0->4], both
    reported at the SAME epoch TE-CCL logs for the merged flow (the
    epoch bookkeeping for the intra-switch relay is already folded
    into TE-CCL's own alpha/beta accounting for that logical step;
    we are recovering the physical edges used, not re-deriving epoch
    timing).
    """
    hops = [int(x) for x in re.findall(r"\d+", switches_str)]
    chain = [src] + hops + [dst]
    return [
        {"chunk": chunk, "chunk_origin": origin,
         "src": chain[i], "dst": chain[i + 1], "epoch": epoch}
        for i in range(len(chain) - 1)
    ]


def parse_flow_strings(lines):
    events, unparsed = [], []
    for s in lines:
        if not isinstance(s, str):
            unparsed.append(repr(s)[:80])
            continue
        m = FLOW_RE.search(s)
        if m:
            origin = int(m.group("origin"))
            src, dst = int(m.group("src")), int(m.group("dst"))
            epoch = int(m.group("epoch"))
            chunk = m.group("chunk")
            switches = m.group("switches")
            if switches:
                events.extend(_expand_switch_hops(
                    origin, src, dst, epoch, chunk, switches))
            else:
                events.append({
                    "chunk": chunk,
                    "chunk_origin": origin,
                    "src": src,
                    "dst": dst,
                    "epoch": epoch,
                })
            continue
        m = LOOSE_RE.search(s)
        if m:
            events.append({
                "chunk": "?",
                "chunk_origin": None,
                "src": int(m.group("src")),
                "dst": int(m.group("dst")),
                "epoch": int(m.group("epoch")),
            })
            continue
        unparsed.append(s[:120])

    if not events:
        raise ValueError(
            "No flow lines parsed. First unparsed lines:\n  "
            + "\n  ".join(unparsed[:5])
        )
    if unparsed:
        # Loud but non-fatal: a few odd lines shouldn't discard a long
        # solve, but silence would hide format drift.
        print(f"  WARNING: {len(unparsed)} flow lines unparsed, "
              f"e.g. {unparsed[0]!r}")
    return events


# ============================================================
# Routes B / C — dict-shaped fallbacks
# ============================================================

CAND_SRC = ("src", "source", "from", "src_node", "sender")
CAND_DST = ("dst", "dest", "destination", "to", "dst_node",
            "receiver")
CAND_EPOCH = ("epoch", "time", "step", "round", "t", "start_epoch")
CAND_CHUNK = ("chunk", "chunk_id", "commodity", "data", "id")


def _first(d, cands):
    for k in cands:
        if k in d:
            return k
    return None


def parse_dict_entries(entries):
    e0 = entries[0]
    ks, kd = _first(e0, CAND_SRC), _first(e0, CAND_DST)
    ke, kc = _first(e0, CAND_EPOCH), _first(e0, CAND_CHUNK)
    if ks is None or kd is None or ke is None:
        raise ValueError(
            f"Dict entries unrecognized: {sorted(e0.keys())}"
        )
    return [{
        "chunk": str(e.get(kc, "?")),
        "chunk_origin": None,
        "src": int(e[ks]),
        "dst": int(e[kd]),
        "epoch": int(round(float(e[ke]))),
    } for e in entries]


# ============================================================
# Main schedule parser
# ============================================================

def parse_schedule(sched_json):
    """Returns (events, meta)."""
    meta = {"epoch_duration_sec": None, "mip_gap": None,
            "solver_status": None, "extra": {}}

    if isinstance(sched_json, list):
        if sched_json and isinstance(sched_json[0], str):
            return parse_flow_strings(sched_json), meta
        return parse_dict_entries(sched_json), meta

    if not isinstance(sched_json, dict):
        raise ValueError(f"Unexpected root: {type(sched_json)}")

    # ---- scalar metadata (numbered prefixes tolerated) ----
    k = find_key(sched_json, "epoch_duration")
    if k is not None:
        meta["epoch_duration_sec"] = to_float(sched_json[k])

    k = find_key(sched_json, "mip_gap", "gap")
    if k is not None:
        meta["mip_gap"] = to_float(sched_json[k])

    k = find_key(sched_json, "solver_status", "status", "optimal")
    if k is not None:
        meta["solver_status"] = str(sched_json[k])[:120]

    for orig, v in sched_json.items():
        if isinstance(v, (int, float, str)):
            meta["extra"][norm_key(orig)] = v

    # ---- route A ----
    k = find_key(sched_json, "flows", "flow")
    if k is not None and isinstance(sched_json[k], list) \
            and sched_json[k]:
        first = sched_json[k][0]
        if isinstance(first, str):
            return parse_flow_strings(sched_json[k]), meta
        if isinstance(first, dict):
            return parse_dict_entries(sched_json[k]), meta

    # ---- route B ----
    for cand in ("schedule", "sends", "transfers", "steps",
                 "solution"):
        k = find_key(sched_json, cand)
        if k is not None and isinstance(sched_json[k], list) \
                and sched_json[k]:
            first = sched_json[k][0]
            if isinstance(first, str):
                return parse_flow_strings(sched_json[k]), meta
            if isinstance(first, dict):
                return parse_dict_entries(sched_json[k]), meta

    # ---- route C: merge any string lists ----
    merged = []
    for v in sched_json.values():
        if isinstance(v, list) and v and isinstance(v[0], str):
            merged.extend(v)
    if merged:
        return parse_flow_strings(merged), meta

    raise ValueError(
        f"Unrecognized schedule layout. Normalized keys: "
        f"{sorted(key_index(sched_json).keys())}"
    )


# ============================================================
# Topology from the INPUT json
# ============================================================

def parse_topology_from_input(input_cfg, input_path):
    """
    Returns (num_nodes, [src_list, dst_list], capacity, latency,
    switch_indices).

    (a) inline matrices in TopologyParams
    (b) sidecar '<input>.topology.json':
        {"num_nodes": N,
         "edges": [{"src":i,"dst":j,"capacity":c,"latency":l}, ...],
         "switch_indices": [...]}
    """
    tp = input_cfg.get("TopologyParams", {})

    cap = tp.get("capacity", tp.get("capacity_matrix"))
    if cap is not None:
        alpha = tp.get("alpha", tp.get("latency",
                                       tp.get("alpha_matrix")))
        n = len(cap)
        src, dst, c, l = [], [], [], []
        for u in range(n):
            for v in range(n):
                cuv = cap[u][v]
                if u != v and cuv and cuv > 0:
                    src.append(u); dst.append(v)
                    c.append(float(cuv))
                    l.append(float(alpha[u][v]) if alpha else 0.0)
        return n, [src, dst], c, l, tp.get("switch_indices", [])

    sidecar = Path(str(input_path) + ".topology.json")
    if sidecar.exists():
        t = json.loads(sidecar.read_text())
        src = [e["src"] for e in t["edges"]]
        dst = [e["dst"] for e in t["edges"]]
        c = [float(e["capacity"]) for e in t["edges"]]
        l = [float(e.get("latency", 0.0)) for e in t["edges"]]
        return (int(t["num_nodes"]), [src, dst], c, l,
                t.get("switch_indices", []))

    raise ValueError(
        f"Cannot extract topology for {input_path.name}. "
        f"TopologyParams keys: {sorted(tp.keys())}. Write a sidecar "
        f"'{sidecar.name}' (num_nodes / edges / switch_indices) — a "
        f"one-time, five-minute job per topology that also retires "
        f"the hardcoded switch ids downstream."
    )


# ============================================================
# Collective codes (teccl/input_data.py uses ints)
# ============================================================

COLLECTIVE_MAP = {1: "AllGather", 2: "AlltoAll", 3: "AllReduce",
                  4: "ReduceScatter", 5: "Broadcast"}


def collective_name(v):
    if isinstance(v, str):
        return v
    return COLLECTIVE_MAP.get(int(v), f"Collective_{v}")


# ============================================================
# Label sanity: AllGather coverage invariant
# ============================================================

def check_allgather_coverage(events, num_nodes, switch_indices=()):
    """
    AllGather is DEFINED by: every source's chunk reaches every other
    participant. A schedule that violates this is not a valid label,
    no matter how well it parsed. Returns a report dict; the caller
    decides whether to warn or refuse.

    Reachability is computed per chunk-origin over the edges the
    schedule actually uses, so multi-hop delivery counts correctly.
    """
    participants = {i for i in range(num_nodes)
                    if i not in set(switch_indices)}

    # edges used, per chunk origin (None = unknown origin)
    per_origin = {}
    for e in events:
        o = e.get("chunk_origin")
        per_origin.setdefault(o, set()).add((e["src"], e["dst"]))

    origins_seen = {o for o in per_origin if o is not None}
    nodes_seen = ({e["src"] for e in events}
                  | {e["dst"] for e in events})

    incomplete = {}
    for o, edges in per_origin.items():
        if o is None:
            continue
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
        "participants": sorted(participants),
        "nodes_seen": sorted(nodes_seen),
        "nodes_absent": sorted(participants - nodes_seen),
        "origins_seen": sorted(origins_seen),
        "origins_absent": sorted(participants - origins_seen),
        "incomplete_origins": incomplete,
        "ok": (not incomplete
               and not (participants - origins_seen)
               and not (participants - nodes_seen)),
    }


# ============================================================
# Driver
# ============================================================

def run_one(template_path, chunk_bytes, store):
    topo_name = template_path.stem
    key = (topo_name, float(chunk_bytes))
    if store.has(key):
        print(f"SKIP {topo_name} @ {chunk_bytes:g}B")
        return

    cfg = json.loads(template_path.read_text())
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{topo_name}_{int(chunk_bytes)}"
    out_sched = WORK_DIR / f"{tag}_schedule.json"

    inst = cfg.setdefault("InstanceParams", {})
    for k in ("chunk_size", "chunk_size_bytes"):
        if k in inst:
            inst[k] = float(chunk_bytes)

    # ABSOLUTE output path: TE-CCL resolves relative paths against the
    # CURRENT WORKING DIRECTORY — that is how stray
    # 'teccl/examples/schedules/' trees appear when you run from
    # another folder. Absolute paths make cwd irrelevant.
    inst["schedule_output_file"] = str(out_sched.resolve())

    run_input = WORK_DIR / f"{tag}_input.json"
    run_input.write_text(json.dumps(cfg, indent=2))

    t0 = time.time()
    proc = subprocess.run(TECCL_CMD + [str(run_input)],
                          capture_output=True, text=True,
                          timeout=TIMEOUT_SEC)
    elapsed = time.time() - t0
    if proc.returncode != 0 or not out_sched.exists():
        raise RuntimeError(
            f"teccl solve failed for {tag}: "
            f"{proc.stderr[-500:] if proc.stderr else 'no schedule'}"
        )

    sched_json = json.loads(out_sched.read_text())
    events, meta = parse_schedule(sched_json)
    n, edge_index, cap, lat, switches = parse_topology_from_input(
        cfg, template_path
    )

    coll = collective_name(inst.get("collective", "AllGather"))

    # Refuse to store a label that violates the collective's own
    # definition — a silently incomplete AllGather would teach the
    # Agent that some nodes need not be served.
    if coll == "AllGather":
        cov = check_allgather_coverage(events, n, switches)
        if not cov["ok"]:
            raise ValueError(
                f"AllGather coverage violated for {tag}: "
                f"absent nodes={cov['nodes_absent']}, "
                f"absent origins={cov['origins_absent']}, "
                f"origins with unreached destinations="
                f"{ {k: v[:4] for k, v in list(cov['incomplete_origins'].items())[:3]} }. "
                f"Resolve the topology/demand mismatch before "
                f"generating labels (see STRICT_COVERAGE)."
            )

    completion_epoch = max(e["epoch"] for e in events)
    epoch_dur = meta["epoch_duration_sec"]
    # Epochs are 0-indexed and the collective finishes at the END of
    # the last used epoch, hence (max_epoch + 1) * duration.
    completion_time = ((completion_epoch + 1) * epoch_dur
                       if epoch_dur else None)

    sample = build_sample(
        topology_name=topo_name,
        collective=coll,
        message_size_bytes=chunk_bytes,
        num_nodes=n,
        edge_index=edge_index,
        capacity=cap, latency=lat,
        capacity_unit="GBps_teccl",
        events=events,
        switch_indices=switches,
        source="teccl",
        solver_wall_time_sec=elapsed,
        mip_gap=meta["mip_gap"],
        solver_status=meta["solver_status"],
        completion_time=completion_time,
        epoch_duration_sec=epoch_dur,
        extra={"teccl_meta": meta["extra"]},
    )
    store.add(sample)
    print(f"OK {tag} | epochs={sample['completion_epoch']} "
          f"| events={len(events)} | epoch_dur={epoch_dur} "
          f"| t_complete={completion_time} | {elapsed:.1f}s "
          f"| saved={len(store.samples)}")


# ============================================================
# Batch extraction of pre-solved outputs (no solver invocation)
# ============================================================

def run_provided_file(sched_path: Path, store):
    """
    Parse one already-solved schedule from experiments/output_provided
    and fold it into the dataset. Path layout gives every piece of
    metadata the schedule JSON itself does not carry (collective,
    chassis, mode, message size) — see the PROVIDED_* config above.
    """
    rel = sched_path.relative_to(PROVIDED_DIR)
    parts = rel.parts  # (TOPO_output, chassis, collective, mode, size.json)
    topo_output, chassis_folder, collective_folder, mode, size_file = parts
    topo = topo_output.replace("_output", "")          # DGX2 | NDv2
    collective = collective_folder                     # AllGather | AlltoAll
    size_bytes = parse_provided_size_bytes(Path(size_file).stem)

    # Fast/Fast_Early_Stop/Slow all share (topology_name, message_size)
    # since topology_name is (topo, chassis)-level, not per-mode — so
    # the store key MUST include collective + mode or these distinct,
    # legitimate samples would collide/skip each other.
    store_key = (f"{topo}_{chassis_folder}", size_bytes, collective, mode)
    if store.has(store_key):
        print(f"SKIP {'_'.join(str(k) for k in store_key)}")
        return

    key = (topo, chassis_folder)
    template_path = PROVIDED_SIDECAR_MAP.get(key)
    if template_path is None:
        raise ValueError(
            f"No sidecar mapping for {key} — add it to "
            f"PROVIDED_SIDECAR_MAP after generating that topology's "
            f"sidecar with make_teccl_sidecar.py."
        )
    cfg = json.loads(template_path.read_text())

    topology_name = f"{topo}_{chassis_folder}"          # e.g. NDv2_4_chassis
    tag = f"{topology_name}_{collective}_{mode}_{Path(size_file).stem}"
    mip_gap = PROVIDED_MIP_GAP.get(mode, PROVIDED_MIP_GAP_DEFAULT)

    sched_json = json.loads(sched_path.read_text())
    events, meta = parse_schedule(sched_json)
    # provided-output mip_gap is a KNOWN solve-time parameter (see
    # PROVIDED_MIP_GAP), not something the schedule JSON reports —
    # prefer it unless the file itself happens to carry one.
    if meta["mip_gap"] is None:
        meta["mip_gap"] = mip_gap

    n, edge_index, cap, lat, switches = parse_topology_from_input(
        cfg, template_path
    )

    coll_name = collective_name(1 if collective == "AllGather" else 2)
    if coll_name == "AllGather":
        cov = check_allgather_coverage(events, n, switches)
        if not cov["ok"]:
            raise ValueError(
                f"AllGather coverage violated for {tag}: "
                f"absent nodes={cov['nodes_absent']}, "
                f"absent origins={cov['origins_absent']}, "
                f"incomplete origins="
                f"{ {k: v[:4] for k, v in list(cov['incomplete_origins'].items())[:3]} }."
            )
    # AlltoAll's invariant is "every ORDERED pair (i,j) is served", not
    # "every source reaches everyone" — our event schema (chunk id +
    # origin, no per-chunk intended destination) can't verify that
    # without richer per-chunk destination info the provided outputs
    # don't carry. We parse AlltoAll schedules but do NOT claim to
    # have validated their coverage; documented, not silently assumed.

    completion_epoch = max(e["epoch"] for e in events)
    epoch_dur = meta["epoch_duration_sec"]
    completion_time = ((completion_epoch + 1) * epoch_dur
                       if epoch_dur else None)

    sample = build_sample(
        topology_name=topology_name,
        collective=coll_name,
        message_size_bytes=size_bytes,
        num_nodes=n,
        edge_index=edge_index,
        capacity=cap, latency=lat,
        capacity_unit="GBps_teccl",
        events=events,
        switch_indices=switches,
        source="teccl",
        solver_wall_time_sec=None,
        mip_gap=meta["mip_gap"],
        solver_status=meta["solver_status"],
        completion_time=completion_time,
        epoch_duration_sec=epoch_dur,
        extra={"teccl_meta": meta["extra"], "provided_mode": mode},
    )
    store.add(sample)
    print(f"OK {tag} | events={len(events)} | epoch_dur={epoch_dur} "
          f"| t_complete={completion_time} | saved={len(store.samples)}")


def run_provided_sweep():
    files = sorted(PROVIDED_DIR.rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files under {PROVIDED_DIR}")
    store = IncrementalStore(
        OUT_PATH,
        ("topology_name", "message_size", "collective", "provided_mode"),
    )
    ok, failed = 0, []
    for f in files:
        try:
            run_provided_file(f, store)
            ok += 1
        except Exception as e:                          # noqa: BLE001
            failed.append((str(f.relative_to(PROVIDED_DIR)), str(e)))
            print(f"FAILED {f.relative_to(PROVIDED_DIR)}: {e}")
    print(f"\nDONE — {ok}/{len(files)} provided files ingested, "
          f"{len(store.samples)} total samples in {OUT_PATH}")
    if failed:
        print(f"\n{len(failed)} failures:")
        for name, err in failed[:20]:
            print(f"  {name}: {err}")


# ============================================================
# Inspect / dry-parse
# ============================================================

def inspect(path):
    obj = json.loads(Path(path).read_text())

    def walk(o, prefix="", depth=0):
        if depth > 3:
            return
        if isinstance(o, dict):
            for k in list(o.keys())[:15]:
                print(f"{prefix}{k}  ->  {type(o[k]).__name__}")
                walk(o[k], prefix + "    ", depth + 1)
        elif isinstance(o, list) and o:
            print(f"{prefix}[list of {len(o)}] first: {o[0]!r}"[:200])
            if isinstance(o[0], dict):
                walk(o[0], prefix + "    ", depth + 1)

    walk(obj)


def dry_parse(path, num_nodes=None, switches=()):
    """Parse an existing schedule JSON without running the solver."""
    obj = json.loads(Path(path).read_text())
    events, meta = parse_schedule(obj)
    epochs = sorted({e["epoch"] for e in events})
    edges = {(e["src"], e["dst"]) for e in events}
    nodes = {e["src"] for e in events} | {e["dst"] for e in events}
    origins = {e["chunk_origin"] for e in events
               if e.get("chunk_origin") is not None}
    chunks = {e["chunk"] for e in events}

    print(f"parsed events      : {len(events)}")
    print(f"distinct edges used: {len(edges)}")
    print(f"node ids seen      : {min(nodes)}..{max(nodes)} "
          f"({len(nodes)} distinct)")
    print(f"chunk ids          : {sorted(chunks)[:10]}"
          f"{' ...' if len(chunks) > 10 else ''} "
          f"({len(chunks)} distinct)")
    if origins:
        print(f"chunk origins      : {min(origins)}..{max(origins)} "
              f"({len(origins)} distinct) -> {sorted(origins)}")
        # per-origin transfer counts expose asymmetric participation
        counts = {}
        for e in events:
            o = e.get("chunk_origin")
            if o is not None:
                counts[o] = counts.get(o, 0) + 1
        print(f"transfers/origin   : "
              f"{ {k: counts[k] for k in sorted(counts)} }")
    else:
        print("chunk origins      : NOT PRESENT in the log format")

    if num_nodes:
        cov = check_allgather_coverage(events, num_nodes, switches)
        print(f"\nAllGather coverage : "
              f"{'OK' if cov['ok'] else 'VIOLATED'}")
        print(f"  nodes absent     : {cov['nodes_absent']}")
        print(f"  origins absent   : {cov['origins_absent']}")
        if cov["incomplete_origins"]:
            print(f"  origins that do not reach everyone: "
                  f"{ {k: v for k, v in list(cov['incomplete_origins'].items())[:5]} }")
    print(f"epochs             : {epochs[0]}..{epochs[-1]} "
          f"({len(epochs)} distinct)")
    print(f"epoch_duration_sec : {meta['epoch_duration_sec']}")
    print(f"mip_gap            : {meta['mip_gap']}")
    print(f"solver_status      : {meta['solver_status']}")
    if meta["epoch_duration_sec"]:
        print(f"=> completion_time : "
              f"{(epochs[-1] + 1) * meta['epoch_duration_sec']:.6g} s")
    print("\nfirst 3 events:")
    for e in events[:3]:
        print(" ", e)
    print("\nscalar metadata found:")
    for k, v in list(meta["extra"].items())[:15]:
        print(f"  {k}: {str(v)[:80]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", type=str, default=None)
    ap.add_argument("--dry-parse", type=str, default=None)
    ap.add_argument("--num-nodes", type=int, default=None,
                    help="enable the AllGather coverage check")
    ap.add_argument("--switches", type=int, nargs="*", default=(),
                    help="switch/relay node ids excluded from the "
                         "AllGather participant set")
    ap.add_argument("--from-provided", action="store_true",
                    help="ingest experiments/output_provided/ "
                         "(pre-solved TE-CCL paper schedules) instead "
                         "of invoking the solver")
    args = ap.parse_args()

    if args.inspect:
        inspect(args.inspect)
        return
    if args.dry_parse:
        dry_parse(args.dry_parse, args.num_nodes, args.switches)
        return
    if args.from_provided:
        run_provided_sweep()
        return

    # Templates whose real data comes from --from-provided (pre-solved
    # paper outputs) are excluded here so the default sweep doesn't
    # redundantly re-solve them with Gurobi.
    provided_templates = {p.name for p in PROVIDED_SIDECAR_MAP.values()}
    templates = [t for t in sorted(TEMPLATE_DIR.glob("*.json"))
                 if not t.name.endswith(".topology.json")
                 and t.name not in provided_templates]
    if not templates:
        raise FileNotFoundError(f"No templates in {TEMPLATE_DIR}")

    store = IncrementalStore(OUT_PATH,
                             ("topology_name", "message_size"))
    for tpl in templates:
        for cb in CHUNK_SIZES_BYTES:
            try:
                run_one(tpl, cb, store)
            except Exception as e:
                print(f"FAILED {tpl.stem} @ {cb:g}B: {e}")

    print(f"\nDONE — {len(store.samples)} samples in {OUT_PATH}")


if __name__ == "__main__":
    main()
