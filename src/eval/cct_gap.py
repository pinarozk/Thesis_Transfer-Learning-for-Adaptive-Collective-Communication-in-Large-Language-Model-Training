"""
cct_gap.py — downstream CCT-gap measurement (the thesis's main result).

For every test-set graph and every trained seed:
  1. Agent inference (load_pred, sched_pred) -- frozen checkpoint.
  2. Decode load predictions into an executable AllGather schedule
     (schedule_bridge.decode_allgather: max-predicted-load broadcast
     tree per source, feasibility-guaranteed).
  3. Run the REAL discrete-event simulator on the decoded schedule to
     get the agent's actual makespan (CCT) -- not a proxy metric.
  4. Compare against the teacher's own recorded completion_time
     (optimality gap = agent_CCT / teacher_CCT).
  5. Stratify by switch-density (graphs with switch_indices vs none;
     switch-adjacent edge fraction) to check whether the switch-edge
     load-prediction weakness (near-zero Pearson r there, see
     eval_agent.py Q2) actually costs anything downstream.
  6. Compare wall-clock: Agent inference+decode time vs teacher's own
     recorded solve time (teacher_wall_time_sec) -- the speed claim.

Requires: agent_best_seed{seed}.pt checkpoints (train_agent.py) and
pyg_agent_test.pt (convert_to_pyg.py). Depends on schedule_bridge.py's
CAPACITY_TO_BPS being calibrated (fixed to 8e9 -- GB/s->bps -- after a
smoke test showed a ~1.6e10x gap ratio at the old placeholder of 1.0;
verified sane, ~2x, once corrected).
"""

import sys
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "src" / "eval"))
sys.path.append(str(BASE_DIR / "src" / "pipeline"))

from eval_agent import load_model, predict_graph, TEST_PYG, LOAD_USED_THRESHOLD  # noqa: E402
from schedule_bridge import (                                                    # noqa: E402
    to_simulator_topology, decode_allgather, run_ground_truth,
)
from src.pipeline.teacher_replay import teacher_sim_cct                          # noqa: E402

SEEDS = [42, 1337, 2024]
EVAL_DIR = BASE_DIR / "src" / "eval"
OUT_JSON = EVAL_DIR / "cct_gap_summary.json"
OUT_CSV = EVAL_DIR / "cct_gap_per_graph.csv"
CANONICAL_TEST = BASE_DIR / "data" / "processed" / "canonical_agent_test.pt"


def load_raw_schedule_lookup():
    """(topology_name, message_size_bytes, occurrence_idx) -> raw_schedule,
    from the canonical (pre-PyG) test set -- raw_schedule is
    intentionally NOT carried onto PyG Data objects (variable-length
    Python structures break batching), so it must be joined back in by
    key, not by list index (index alignment between canonical and PyG
    lists is not something to assume silently).

    occurrence_idx (NOT variant_id): TE-CCL never sets variant_id or
    link_variant_seed (both None -> both collapse to the same -1
    sentinel), so EVERY DGX2 sample at a given message_size shared one
    lookup key -- 4 of every 5 DGX2 (topology_name, message_size)
    groups got replayed against an arbitrary OTHER sample's schedule
    (confirmed: teacher_sim_cct was bit-identical across all 5 "distinct"
    1GB DGX2 rows despite wildly different teacher_completion_time and
    event counts spanning 1935 to 16033). occurrence_idx instead counts
    position within each (topology_name, message_size) group, in
    canonical list order -- valid because convert_to_pyg.py preserves
    that order 1:1 (appends in a single forward pass, skips only on a
    real per-sample error, and this dataset converts with 0 such
    errors), so building the SAME per-group counter while iterating
    test_data in main() reproduces the identical pairing."""
    canonical = torch.load(CANONICAL_TEST, map_location="cpu",
                          weights_only=False)
    lut = {}
    occurrence = defaultdict(int)
    for s in canonical:
        group = (str(s["topology_name"]), float(s["message_size_bytes"]))
        key = (group[0], group[1], occurrence[group])
        occurrence[group] += 1
        lut[key] = s["raw_schedule"]
    return lut


def switch_edge_fraction(data):
    names = data.edge_feature_names
    if "is_switch_edge" not in names:
        return 0.0
    i = names.index("is_switch_edge")
    return float((data.edge_attr[:, i] > 0.5).float().mean())


def run_for_seed(seed, test_data, teacher_sim_cache, device="cpu"):
    model, threshold, load_target_mode = load_model(seed, device)
    rows = []
    for idx, data in enumerate(test_data):
        t0 = time.perf_counter()
        load_pred, sched = predict_graph(model, data, device,
                                         load_target_mode=load_target_mode)
        topo = to_simulator_topology(data)
        policy, stats = decode_allgather(data, load_pred, sched, threshold)
        infer_decode_time = time.perf_counter() - t0

        row = {
            "seed": seed,
            "idx": idx,
            "topology_name": str(data.topology_name),
            "source_domain": str(data.source_domain),
            "message_size_bytes": float(data.message_size_bytes),
            "has_switch": bool(len(data.switch_indices or []) > 0),
            "switch_edge_frac": switch_edge_fraction(data),
            "num_nodes": int(data.x.size(0)),
            "num_edges": int(data.edge_index.size(1)),
            "selected_edges": stats["selected_edges"],
            "fallback_edges": stats["fallback_edges"],
            "teacher_completion_time": float(data.completion_time),
            "teacher_wall_time_sec": float(data.teacher_wall_time_sec),
            "mip_gap": float(data.mip_gap),
            "agent_infer_decode_time_sec": infer_decode_time,
        }

        t_cct, t_stats = teacher_sim_cache[idx]
        row["teacher_sim_cct"] = t_cct
        row["teacher_replay_missing_arrival"] = (
            t_stats["missing_arrival"] if t_stats else None)
        row["teacher_replay_ok"] = t_cct is not None

        try:
            agent_cct = run_ground_truth(topo, policy)
            row["agent_cct"] = agent_cct
            row["gap_ratio"] = (agent_cct / row["teacher_completion_time"]
                                 if row["teacher_completion_time"] > 0
                                 else float("nan"))
            # gap_ratio_sim: BOTH sides measured in the same simulator
            # with the same topology conversion (see teacher_replay.py)
            # -- the thesis-table metric. gap_ratio (above) is kept for
            # reference/comparison only.
            row["gap_ratio_sim"] = (
                agent_cct / t_cct
                if (t_cct is not None and t_cct > 0) else float("nan"))
            row["sim_ok"] = True
        except Exception as e:
            row["agent_cct"] = float("nan")
            row["gap_ratio"] = float("nan")
            row["gap_ratio_sim"] = float("nan")
            row["sim_ok"] = False
            row["sim_error"] = str(e)

        if row["teacher_wall_time_sec"] > 0 and not np.isnan(row["teacher_wall_time_sec"]):
            row["speedup_vs_teacher"] = (row["teacher_wall_time_sec"]
                                          / infer_decode_time)
        else:
            row["speedup_vs_teacher"] = float("nan")

        rows.append(row)
    return rows


def summarize(df):
    ok = df[df["sim_ok"]]
    summary = {
        "n_graphs": int(len(df)),
        "n_simulated_ok": int(len(ok)),
        "n_simulator_errors": int((~df["sim_ok"]).sum()),
    }

    def block(sub, label):
        if len(sub) == 0:
            return {}
        out = {f"{label}_n": int(len(sub))}
        for col, tag in [("gap_ratio", "gap_ratio"),
                        ("gap_ratio_sim", "gap_ratio_sim")]:
            gr = sub[col].dropna()
            out.update({
                f"{label}_{tag}_mean": float(gr.mean()) if len(gr) else float("nan"),
                f"{label}_{tag}_median": float(gr.median()) if len(gr) else float("nan"),
                f"{label}_{tag}_std": float(gr.std()) if len(gr) else float("nan"),
                f"{label}_{tag}_min": float(gr.min()) if len(gr) else float("nan"),
                f"{label}_{tag}_max": float(gr.max()) if len(gr) else float("nan"),
            })
        return out

    summary["overall"] = block(ok, "overall")

    for seed in sorted(df["seed"].unique()):
        summary[f"seed_{seed}"] = block(ok[ok["seed"] == seed], "seed")

    for has_sw in [True, False]:
        label = "switch_topologies" if has_sw else "switchless_topologies"
        summary[label] = block(ok[ok["has_switch"] == has_sw], label)

    for src in sorted(df["source_domain"].unique()):
        summary[f"source_{src}"] = block(ok[ok["source_domain"] == src], f"source_{src}")

    for topo in sorted(df["topology_name"].unique()):
        summary[f"topo_{topo}"] = block(ok[ok["topology_name"] == topo], f"topo_{topo}")

    mis = ok["teacher_replay_missing_arrival"].dropna()
    summary["teacher_replay_sanity"] = {
        "n_missing_arrival_nonzero": int((mis > 0).sum()),
        "max_missing_arrival": int(mis.max()) if len(mis) else 0,
    }

    speed = ok["speedup_vs_teacher"].dropna()
    summary["speed"] = {
        "n": int(len(speed)),
        "speedup_mean": float(speed.mean()) if len(speed) else float("nan"),
        "speedup_median": float(speed.median()) if len(speed) else float("nan"),
        "mean_agent_infer_decode_time_sec": float(ok["agent_infer_decode_time_sec"].mean()),
        "mean_teacher_wall_time_sec": float(ok["teacher_wall_time_sec"].replace(
            [np.inf, -np.inf], np.nan).dropna().mean()),
    }
    return summary


def main():
    test_data = torch.load(TEST_PYG, map_location="cpu", weights_only=False)
    print(f"Loaded {len(test_data)} test graphs.")

    print("Replaying teacher schedules in the real simulator "
          "(seed-independent, computed once per graph)...")
    raw_schedule_lut = load_raw_schedule_lookup()
    teacher_sim_cache = {}
    n_replay_err = 0
    # Same occurrence-index scheme as load_raw_schedule_lookup(): count
    # position within each (topology_name, message_size) group as we
    # walk test_data in order, which convert_to_pyg.py guarantees
    # matches the canonical list's own order 1:1 for this dataset (0
    # conversion errors) -- variant_id/link_variant_seed do NOT
    # disambiguate here (TE-CCL leaves both None -> both sentinel to
    # the same -1, silently colliding every DGX2 sample sharing a
    # message_size onto one arbitrary schedule; caught by teacher_sim_cct
    # being bit-identical across 5 "distinct" DGX2 rows with wildly
    # different teacher_completion_time and event counts).
    occurrence = defaultdict(int)
    for idx, data in enumerate(test_data):
        group = (str(data.topology_name), float(data.message_size_bytes))
        key = (group[0], group[1], occurrence[group])
        occurrence[group] += 1
        print(f"  [{idx+1}/{len(test_data)}] replaying {key} "
              f"(n_events={len(raw_schedule_lut.get(key) or [])}) ...",
              flush=True)
        t0 = time.perf_counter()
        sched = raw_schedule_lut.get(key)
        if sched is None:
            print(f"  WARNING: no raw_schedule match for {key}, "
                  f"teacher replay skipped for this graph")
            teacher_sim_cache[idx] = (None, None)
            n_replay_err += 1
            continue
        try:
            t_cct, t_stats = teacher_sim_cct(
                data, sched, float(data.message_size_bytes))
            teacher_sim_cache[idx] = (t_cct, t_stats)
            print(f"    -> done in {time.perf_counter()-t0:.1f}s, "
                  f"t_cct={t_cct:.6g}, stats={t_stats}", flush=True)
        except Exception as e:
            print(f"  WARNING: teacher replay failed for {key}: {e}")
            teacher_sim_cache[idx] = (None, None)
            n_replay_err += 1
    print(f"Teacher replay done: {len(test_data) - n_replay_err}/"
          f"{len(test_data)} ok")

    all_rows = []
    for seed in SEEDS:
        print(f"--- seed {seed} ---")
        rows = run_for_seed(seed, test_data, teacher_sim_cache)
        n_err = sum(1 for r in rows if not r["sim_ok"])
        print(f"  {len(rows)} graphs, {n_err} simulator errors")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV)

    summary = summarize(df)
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved:", OUT_JSON)
    print(json.dumps(summary["overall"], indent=2))
    print(json.dumps(summary["speed"], indent=2))


if __name__ == "__main__":
    main()
