"""
make_teccl_sidecar.py — turn a TE-CCL topology into a physical-units
sidecar for extract_teccl.py.

WHY PHYSICAL UNITS (the decision this script encodes):
TE-CCL's internal capacity matrix is expressed in CHUNKS PER EPOCH,
i.e. entry = conversion(beta) / chunk_size. That is the right unit for
its own MILP but the wrong unit for us:

  * our canonical edge feature log10_tx_cost = log10(msg / capacity)
    would become chunk_size-dependent — the SAME physical link would
    get different features at different message sizes, injecting a
    spurious correlation into exactly the message-size analysis (Q1)
    we intend to run;
  * physical units are what makes TE-CCL commensurable with your own
    ILP and with the simulator (CAPACITY_TO_BPS becomes well-defined
    per source);
  * being chunk_size-independent, ONE sidecar per topology serves the
    whole message-size sweep.

UNIT CONVERSION (from the beta comments in teccl/topologies/*.py):
    beta is in microseconds per MB, so
        rate [GB/s] = 1e6 bytes / (beta * 1e-6 s) / 1e9
                    = 1000 / beta
    beta = 23  -> 43.478 GB/s
    beta = 46  -> 21.739 GB/s
    beta = 105 ->  9.524 GB/s   (105, not the 107 routing-threshold
                                 approximation)
alpha (latency) in TE-CCL is already in seconds and chunk_size
independent, so it is copied as-is.

HOW IT WORKS
1. Instantiate the topology through TE-CCL itself (so we get exactly
   the matrix the solver used — no hand transcription).
2. Collect the distinct nonzero capacity entries.
3. Map each distinct entry to a physical GB/s rate: entries are
   proportional to 1/beta, so sorting entries descending and matching
   against the sorted physical rates is exact whenever the number of
   distinct link classes matches. The mapping is PRINTED for you to
   verify, and can be overridden explicitly via RAW_TO_GBPS.
4. Write '<input>.topology.json' next to the input template.

USAGE
    python make_teccl_sidecar.py \
        --input teacher_inputs/teccl/ndv2_input.json \
        --module ndv2                    # teccl/topologies/ndv2.py
    # optional: --class NDv2  --switches 0  --dry-run
"""

import argparse
import importlib
import inspect as pyinspect
import json
from pathlib import Path

# Physical rates are PER-TOPOLOGY-MODULE, not universal — NDv2 and
# DGX2 derive their capacity matrices from unrelated hardware
# constants. Reusing one module's rate list for another silently
# produces numerically wrong physical values (caught exactly this way
# on DGX2: auto-mapping borrowed NDv2's beta-derived rates and gave
# 43.478/21.739 GB/s instead of the correct 125/12.5 GB/s implied by
# dgx2.py's own `125 / chunk_size` / `12.5 / chunk_size` constants).
#
# NDv2 (teccl/topologies/ndv2.py): beta is documented in us/MB in the
# source comments -> rate[GB/s] = 1000 / beta.
#   beta = 23, 46, 105 -> 43.478, 21.739, 9.524 GB/s
#
# DGX2 (teccl/topologies/dgx2.py): constants are NOT documented as a
# beta; they are used directly as `constant / chunk_size`, so the
# constant IS the physical GB/s at chunk_size=1 (same convention as
# NDv2, verified numerically: 125/0.0625=2000 and 12.5/0.0625=200
# match the raw matrix entries exactly) — CONFIDENCE: pattern-matched
# from code structure, not from an explicit hardware-spec comment
# like NDv2's; flagged as such in the sidecar's "note" field.
MODULE_PHYSICAL_GBPS = {
    "ndv2": sorted((1000.0 / b for b in (23.0, 46.0, 105.0)),
                   reverse=True),                    # [43.478, 21.739, 9.524]
    "dgx2": [125.0, 12.5],
}
MODULE_CONFIDENCE = {
    "ndv2": "verified: explicit beta[us/MB] comment in source",
    "dgx2": "unverified_pattern_match: constant/chunk_size assumed "
            "== physical GB/s at chunk_size=1, same convention as "
            "ndv2 but no explicit hardware-spec comment in dgx2.py",
}

# Explicit override: {raw_matrix_entry: physical_GBps}. Leave empty to
# use the automatic proportional mapping (which is printed anyway).
RAW_TO_GBPS = {}


# ============================================================
# Topology instantiation
# ============================================================

def load_topology(module_name, class_name, topo_params):
    """
    Import teccl.topologies.<module_name> and instantiate the topology
    class exactly the way teccl/cli/solve.py does it: start from a
    default TopologyParams() and setattr each JSON key onto it (this
    is why extra keys like "option" don't break construction — they
    just become inert attributes, same as in the real solve path).
    """
    from teccl.input_data import TopologyParams  # type: ignore

    mod = importlib.import_module(f"teccl.topologies.{module_name}")

    candidates = [
        obj for name, obj in vars(mod).items()
        if pyinspect.isclass(obj) and obj.__module__ == mod.__name__
    ]
    if class_name:
        candidates = [c for c in candidates
                      if c.__name__ == class_name]
    if not candidates:
        raise SystemExit(
            f"No topology class found in {mod.__name__}. "
            f"Classes present: "
            f"{[c.__name__ for c in vars(mod).values() if pyinspect.isclass(c)]}"
            f"  — pass --class explicitly."
        )
    if len(candidates) > 1:
        print(f"NOTE: several classes found "
              f"({[c.__name__ for c in candidates]}); using the first."
              f" Override with --class.")
    cls = candidates[0]

    tp = TopologyParams()
    for k, v in topo_params.items():
        setattr(tp, k, v)
    return cls(tp)


def get_matrix(topo, *names):
    for n in names:
        m = getattr(topo, n, None)
        if m is not None:
            return m, n
    raise SystemExit(
        f"None of {names} found on the topology object. "
        f"Attributes: {[a for a in dir(topo) if not a.startswith('_')]}"
    )


def flatten_epochs(mat, label):
    """TE-CCL supports per-epoch capacity matrices (variable
    bandwidth). If we got a 3-D structure, use epoch 0 and say so."""
    if (isinstance(mat, list) and mat and isinstance(mat[0], list)
            and mat[0] and isinstance(mat[0][0], list)):
        print(f"NOTE: {label} is per-epoch (3-D); using epoch 0. "
              f"If your run uses time-varying bandwidth, the sidecar "
              f"records the epoch-0 snapshot.")
        return mat[0]
    return mat


# ============================================================
# Unit recovery
# ============================================================

def build_rate_map(distinct_raw, module_name):
    """raw matrix entry -> physical GB/s"""
    if RAW_TO_GBPS:
        missing = set(distinct_raw) - set(RAW_TO_GBPS)
        if missing:
            raise SystemExit(f"RAW_TO_GBPS is missing {missing}")
        return dict(RAW_TO_GBPS)

    physical_gbps = MODULE_PHYSICAL_GBPS.get(module_name)
    if physical_gbps is None:
        raise SystemExit(
            f"No known physical rate table for module '{module_name}'. "
            f"Known modules: {sorted(MODULE_PHYSICAL_GBPS)}. Add an "
            f"entry to MODULE_PHYSICAL_GBPS derived from that module's "
            f"own capacity-construction code (do NOT reuse another "
            f"topology's rates — they come from unrelated hardware "
            f"constants), or pass --rate-map explicitly."
        )
    print(f"\nUsing physical rate table for '{module_name}' "
          f"({MODULE_CONFIDENCE.get(module_name, 'unknown confidence')})")

    raw_sorted = sorted(distinct_raw, reverse=True)
    if len(raw_sorted) > len(physical_gbps):
        raise SystemExit(
            f"{len(raw_sorted)} distinct capacity values "
            f"{raw_sorted} but only {len(physical_gbps)} known link "
            f"classes {physical_gbps} for module '{module_name}'. Fill "
            f"RAW_TO_GBPS explicitly from that module's own source."
        )
    mapping = {r: g for r, g in zip(raw_sorted, physical_gbps)}

    # sanity: entries should be proportional to the physical rates
    base_r, base_g = raw_sorted[0], mapping[raw_sorted[0]]
    print("\nRate mapping (VERIFY against the beta comments):")
    for r in raw_sorted:
        expected = base_g * (r / base_r)
        flag = "" if abs(expected - mapping[r]) / mapping[r] < 0.08 \
            else "   <-- ratio mismatch, check manually"
        print(f"  raw {r:<12.6g} -> {mapping[r]:8.3f} GB/s "
              f"(proportional estimate {expected:8.3f}){flag}")
    return mapping


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="TE-CCL input JSON (template)")
    ap.add_argument("--module", required=True,
                    help="module under teccl.topologies, e.g. ndv2")
    ap.add_argument("--class", dest="cls", default=None)
    ap.add_argument("--switches", type=int, nargs="*", default=None,
                    help="switch node ids (default: from topology, "
                         "else none)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    input_path = Path(args.input)
    cfg = json.loads(input_path.read_text())
    topo_params = cfg.get("TopologyParams", {})
    print(f"TopologyParams: {topo_params}")

    topo = load_topology(args.module, args.cls, topo_params)

    cap, cap_name = get_matrix(topo, "capacity", "capacity_matrix")
    alpha, alpha_name = get_matrix(topo, "alpha", "latency",
                                   "alpha_matrix")
    cap = flatten_epochs(cap, cap_name)
    alpha = flatten_epochs(alpha, alpha_name)

    n = len(cap)
    print(f"topology: {type(topo).__name__}, {n} nodes "
          f"(capacity='{cap_name}', latency='{alpha_name}')")

    distinct = sorted({float(cap[i][j])
                       for i in range(n) for j in range(n)
                       if i != j and cap[i][j] and cap[i][j] > 0})
    rate_map = build_rate_map(distinct, args.module)

    edges = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c = cap[i][j]
            if not c or c <= 0:
                continue
            edges.append({
                "src": i, "dst": j,
                "capacity": round(rate_map[float(c)], 6),
                "latency": float(alpha[i][j]) if alpha else 0.0,
                "raw_capacity_chunks_per_epoch": float(c),
            })

    if args.switches is not None:
        switches = sorted(args.switches)
    else:
        switches = sorted(getattr(topo, "switch_indices", []) or [])

    # isolated nodes: the concrete test for the "missing node" anomaly
    degree = {i: 0 for i in range(n)}
    for e in edges:
        degree[e["src"]] += 1
        degree[e["dst"]] += 1
    isolated = [i for i, d in degree.items() if d == 0]

    print(f"\nedges: {len(edges)} directed")
    print(f"switch_indices: {switches}")
    print(f"ISOLATED NODES (no links at all): {isolated}")
    if isolated:
        print("  ^ these nodes cannot participate in any collective; "
              "this fully explains any node missing from the "
              "schedule log.")

    sidecar = {
        "num_nodes": n,
        "edges": edges,
        "switch_indices": switches,
        "isolated_nodes": isolated,
        "capacity_unit": "GBps",
        "latency_unit": "s",
        "source_topology": {
            "module": args.module,
            "class": type(topo).__name__,
            "topology_params": topo_params,
        },
        "capacity_confidence": MODULE_CONFIDENCE.get(
            args.module, "unknown"),
        "note": ("capacity converted from TE-CCL chunks/epoch to "
                 "physical GB/s using this module's own rate table "
                 "(MODULE_PHYSICAL_GBPS); raw values kept per edge for "
                 "traceability. See capacity_confidence for how firmly "
                 "these physical values are established."),
    }

    out = Path(str(input_path) + ".topology.json")
    if args.dry_run:
        print("\n--dry-run: not writing. First 3 edges:")
        for e in edges[:3]:
            print(" ", e)
    else:
        out.write_text(json.dumps(sidecar, indent=2))
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
