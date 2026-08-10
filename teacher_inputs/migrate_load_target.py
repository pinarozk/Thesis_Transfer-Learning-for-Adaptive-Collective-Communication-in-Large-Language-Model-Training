"""
One-time migration: add load_target to existing raw teacher datasets
by recomputing events_to_targets() from each sample's already-stored
"schedule" field. No re-solving needed -- see teacher_common.py's
updated events_to_targets() for the src/dst vs from/to+amount handling
this depends on.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.teacher_common import events_to_targets  # noqa: E402

FILES = [
    ROOT / "data" / "processed" / "TE_CCL_dataset.pt",
    ROOT / "data" / "processed" / "allgather_teacher_dataset_incremental.pt",
    ROOT / "data" / "processed" / "syccl_dataset.pt",
]

for path in FILES:
    data = torch.load(path, weights_only=False)
    migrated, skipped = 0, 0
    for s in data:
        if "schedule" not in s or "load_target" in s:
            skipped += 1
            continue
        num_nodes = int(s["node_feat"].shape[0])
        routing, load, scheduling, completion_epoch = events_to_targets(
            s["schedule"], num_nodes
        )
        # sanity: routing must match what's already stored (same
        # events, same edge extraction) -- if not, something about
        # the from/to vs src/dst field-priority logic disagrees with
        # how this sample's target was originally built.
        if not torch.equal((routing > 0), (s["routing_target"] > 0)):
            raise ValueError(
                f"{path.name}: recomputed routing_target disagrees "
                f"with stored one for topology "
                f"{s.get('topology_name')} -- do not trust load_target "
                f"here without investigating."
            )
        s["load_target"] = load
        migrated += 1
    torch.save(data, path)
    print(f"{path.name}: migrated {migrated}, skipped {skipped} "
          f"(no schedule / already migrated)")
