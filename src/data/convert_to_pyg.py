"""
Canonical dict dataset -> PyTorch Geometric Data objects.

Fixes vs. previous version
--------------------------
1. Carries the new fields required for correct training:
     - sched_mask            (loss masking on used edges only)
     - sched_norm_factor     (inverse transform of scheduling target)
     - group_key             (group-aware anything downstream)
     - mip_gap / teacher_wall_time_sec (label quality weighting,
       honest ILP-vs-Agent speedup denominators)
2. Loads split-level stats (pos_weight etc.) and re-saves them next to
   the PyG files so the training script has a single source of truth.
3. Stronger consistency assertions (mask/target coherence), and skipped
   samples RAISE at the end if any exist — silently dropping samples
   from a ~100-sample dataset is not acceptable.
"""

import math
from pathlib import Path
from collections import Counter

import torch
from torch_geometric.data import Data


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"

IN_FULL = DATA_DIR / "canonical_agent_dataset.pt"
IN_TRAIN = DATA_DIR / "canonical_agent_train.pt"
IN_VAL = DATA_DIR / "canonical_agent_val.pt"
IN_TEST = DATA_DIR / "canonical_agent_test.pt"
IN_STATS = DATA_DIR / "canonical_agent_stats.pt"

OUT_FULL = DATA_DIR / "pyg_agent_dataset.pt"
OUT_TRAIN = DATA_DIR / "pyg_agent_train.pt"
OUT_VAL = DATA_DIR / "pyg_agent_val.pt"
OUT_TEST = DATA_DIR / "pyg_agent_test.pt"
OUT_STATS = DATA_DIR / "pyg_agent_stats.pt"


def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    data = torch.load(path, weights_only=False)
    if not isinstance(data, list):
        raise TypeError(f"Expected list, got {type(data)} from {path}")
    return data


def dict_to_pyg(sample):
    data = Data(
        x=sample["node_feat"].float(),
        edge_index=sample["edge_index"].long(),
        edge_attr=sample["edge_attr"].float(),

        y_routing=sample["routing_edge_target"].float(),
        y_load=sample["load_edge_target"].float(),
        y_load_residual=sample["load_residual_target"].float(),
        sp_load_log1p=sample["sp_load_latency_log1p"].float(),
        y_scheduling=sample["scheduling_edge_target"].float(),
        sched_mask=sample["scheduling_mask"].float(),

        # unsqueeze(0): PyG batches unrecognized keys by concatenating
        # along dim 0 by default (its node/edge-feature heuristic). A
        # bare [global_dim] vector gets flattened across the batch
        # into [batch_size*global_dim] instead of stacked into
        # [batch_size, global_dim] -- caught by the training-loader
        # smoke test (global_in_dim showed up as 32, not 4, at
        # batch_size=8). The leading dim of size 1 makes it batch like
        # any other per-graph tensor.
        u=sample["u"].float().unsqueeze(0),
    )

    # ---- scalars needed at train/eval time ----
    data.sched_norm_factor = float(sample["scheduling_norm_factor"])
    data.group_key = sample["group_key"]

    # PyG's Data silently DROPS an attribute set to None from its
    # internal mapping (setattr(d, k, None) never lands in
    # d._store._mapping) -- fields that are None for some teacher
    # domains but real values for others (mip_gap: teccl/ilp have it,
    # syccl doesn't; chassis: only teccl sets it; ...) therefore end
    # up with a DIFFERENT key set per Data object. Batch.from_data_list
    # requires identical keys across every object in a batch, so this
    # surfaces as a KeyError the moment two such samples land in the
    # same batch -- caught by a real training-loader smoke test, not
    # by any single-sample check. Fix: always store a real value,
    # using a domain-neutral sentinel (NaN for floats, -1 for ids,
    # "" for strings) instead of None, so every key exists everywhere.
    def _f(v):
        return float(v) if v is not None else float("nan")

    def _i(v):
        return int(v) if v is not None else -1

    def _s(v):
        return v if v is not None else ""

    # ---- teacher quality ----
    data.mip_gap = _f(sample.get("mip_gap"))
    data.teacher_wall_time_sec = _f(sample.get("teacher_wall_time_sec"))
    data.solver_status = _s(sample.get("solver_status"))

    # ---- provenance metadata ----
    data.source_domain = _s(sample.get("source_domain"))
    data.source = _s(sample.get("source"))
    data.topology_name = _s(sample.get("topology_name"))
    data.collective = _s(sample.get("collective"))
    data.mode = _s(sample.get("mode"))
    data.chassis = _i(sample.get("chassis"))
    data.variant_id = _i(sample.get("variant_id"))
    data.link_variant_seed = _i(sample.get("link_variant_seed"))

    data.message_size_raw = _f(sample.get("message_size_raw"))
    data.message_size_bytes = float(sample.get("message_size_bytes"))

    data.num_nodes_meta = int(sample.get("num_nodes"))
    data.num_edges_meta = int(sample.get("num_edges"))
    data.switch_indices = sample.get("switch_indices") or []

    # ---- schema labels ----
    data.node_feature_names = sample.get("node_feature_names")
    data.edge_feature_names = sample.get("edge_feature_names")
    data.global_feature_names = sample.get("global_feature_names")
    data.target_names = sample.get("target_names")

    # ---- optional extras (always set, same sentinel reasoning) ----
    data.completion_epoch = _i(sample.get("completion_epoch"))
    data.completion_time = _f(sample.get("completion_time"))

    # NOTE: raw_schedule is intentionally NOT attached to the PyG object
    # (variable-length Python structures break PyG batching). It stays in
    # the canonical dataset; retrieve it via group_key + variant_id +
    # message_size when decoding or training SimNet.

    return data


def validate(data: Data):
    assert data.x.dim() == 2
    assert data.edge_index.dim() == 2
    assert data.edge_attr.dim() == 2
    assert data.y_routing.dim() == 1
    assert data.y_scheduling.dim() == 1
    assert data.sched_mask.dim() == 1
    assert data.u.dim() == 2 and data.u.shape[0] == 1

    num_edges = data.edge_index.shape[1]
    assert data.edge_attr.shape[0] == num_edges
    assert data.y_routing.shape[0] == num_edges
    assert data.y_load.shape[0] == num_edges
    assert data.y_scheduling.shape[0] == num_edges
    assert data.sched_mask.shape[0] == num_edges

    # mask/target coherence
    used = data.sched_mask.bool()
    assert torch.equal(used, data.y_routing.bool()), \
        "sched_mask must equal positive routing targets"
    assert (data.y_scheduling[used] > 0).all(), \
        "used edges must have strictly positive scheduling targets"
    assert (data.y_scheduling[~used] == 0).all(), \
        "unused edges must have zero scheduling targets"
    assert (data.y_load[used] > 0).all(), \
        "used edges must have strictly positive load targets"
    assert (data.y_load[~used] == 0).all(), \
        "unused edges must have zero load targets"

    assert torch.isfinite(data.x).all()
    assert torch.isfinite(data.edge_attr).all()
    assert torch.isfinite(data.u).all()
    assert torch.isfinite(data.y_load).all()


def convert_split(input_path, output_path, split_name):
    raw = load_dataset(input_path)

    pyg_data, errors = [], []

    for i, sample in enumerate(raw):
        try:
            d = dict_to_pyg(sample)
            validate(d)
            pyg_data.append(d)
        except Exception as e:
            errors.append((i, sample.get("topology_name"), str(e)))

    torch.save(pyg_data, output_path)

    print("\n" + "=" * 70)
    print(split_name)
    print("-" * 70)
    print("input:", input_path)
    print("output:", output_path)
    print("raw samples:", len(raw))
    print("pyg samples:", len(pyg_data))

    if errors:
        print("ERRORS:")
        for item in errors:
            print(" ", item)
        # A dataset this small cannot afford silent drops.
        raise RuntimeError(
            f"{len(errors)} samples failed conversion in {split_name}; "
            f"fix upstream instead of silently dropping."
        )

    summarize_pyg(pyg_data)
    return pyg_data


def summarize_pyg(data):
    if not data:
        print("No data.")
        return

    print("node feature dims:", Counter(d.x.shape[1] for d in data))
    print("edge feature dims:", Counter(d.edge_attr.shape[1] for d in data))
    print("global feature dims:", Counter(d.u.shape[0] for d in data))
    print("node counts:", Counter(d.x.shape[0] for d in data))
    print("topologies:", Counter(d.topology_name for d in data))
    print("source domains:", Counter(d.source_domain for d in data))
    print("group keys:", Counter(d.group_key for d in data))

    y_all = torch.cat([d.y_routing for d in data])
    print("routing positive ratio:", round(float(y_all.mean()), 4),
          "(diagnostic only -- near-degenerate, see load below)")

    load_all = torch.cat([d.y_load for d in data])
    used_all = torch.cat([d.sched_mask for d in data]).bool()
    print("load_edge_target (log1p): mean=%.4f std=%.4f "
          "| used-only: mean=%.4f std=%.4f"
          % (load_all.mean(), load_all.std(),
             load_all[used_all].mean() if used_all.any() else float("nan"),
             load_all[used_all].std() if used_all.any() else float("nan")))

    gaps = [g for d in data
           if not math.isnan(g := getattr(d, "mip_gap", float("nan")))]
    if gaps:
        print("mip_gap: n=%d mean=%.4f max=%.4f"
              % (len(gaps), sum(gaps) / len(gaps), max(gaps)))

    first = data[0]
    print("\nExample Data object:", first)
    print("feature labels:")
    print("  node:", first.node_feature_names)
    print("  edge:", first.edge_feature_names)
    print("  global:", first.global_feature_names)


def main():
    convert_split(IN_FULL, OUT_FULL, "FULL")
    train = convert_split(IN_TRAIN, OUT_TRAIN, "TRAIN")
    convert_split(IN_VAL, OUT_VAL, "VAL")
    convert_split(IN_TEST, OUT_TEST, "TEST")

    # Single source of truth for training hyperparams derived from data.
    if IN_STATS.exists():
        stats = torch.load(IN_STATS, weights_only=False)
    else:
        # Recompute pos_weight from the PyG train split as fallback.
        y = torch.cat([d.y_routing for d in train])
        n_pos = float(y.sum())
        stats = {
            "routing_pos_weight": (y.numel() - n_pos) / max(1.0, n_pos),
            "routing_positive_ratio": n_pos / y.numel(),
        }
    torch.save(stats, OUT_STATS)

    print("\n" + "=" * 70)
    print("DONE")
    print("Saved PyG datasets + stats:")
    for p in (OUT_FULL, OUT_TRAIN, OUT_VAL, OUT_TEST, OUT_STATS):
        print(" ", p)
    print("\nTraining reminders:")
    print("  - BCEWithLogitsLoss(pos_weight=stats['routing_pos_weight'])")
    print("  - scheduling loss: ((pred - y_scheduling)**2 * sched_mask)"
          ".sum() / sched_mask.sum()")


if __name__ == "__main__":
    main()
