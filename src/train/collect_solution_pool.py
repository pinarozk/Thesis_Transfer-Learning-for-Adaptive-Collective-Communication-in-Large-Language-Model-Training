"""
collect_solution_pool.py — closes the Step 1 / Step 2 gap.

The slide requires (Step 1, bullet 3): "Save intermediate solutions
(high / medium / low quality) for SimNet", and (Step 2, bullet 2):
"Training data = {ILP optimum + Agent intermediate solutions}".
Neither existed in the pipeline until now. This script produces the
Agent-solution half and provides the hook for the ILP half.

Quality tiers WITHOUT retraining (design decision, defend in thesis):
instead of checkpoint snapshots during BC training (which entangles
'solution quality' with 'training epoch' and requires re-running
training), quality is controlled by degrading the FINAL Agent's edge
probabilities in a graded, deterministic way:

    high   : greedy decode at the tuned threshold
             (the Agent's best answer)
    medium : masks sampled from Bernoulli(p)
             (the Agent's own uncertainty realized)
    low    : Bernoulli(alpha*p + (1-alpha)*Uniform), alpha=0.4
             (heavily noised — plausible-but-poor schedules)

Every decoded schedule is executed in the REAL simulator, so labels
are ground truth. Output samples use the exact raw-sample schema of
generate_simnet_dataset.py, with policy_type in
{agent_high, agent_medium, agent_low} — build_simnet_pyg.py merges
them with a one-line RAW_PATHS addition (see bottom of this file),
after which the policy-type breakdown in eval_simnet.py automatically
reports surrogate accuracy on each quality tier: the slide's "learns
the full distribution of solution quality", now measurable.

ILP-OPTIMUM SCHEDULES: canonical samples preserve `raw_schedule`
exactly for this purpose. Converting ILP schedule events into
PolicyEntry lists requires knowing the event format produced by
solve_allgather_ilp; implement `ilp_schedule_to_policy()` below (a
NotImplementedError guards it) and set INCLUDE_ILP = True.
"""

import sys
import math
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.agent_gnn import StrongAgentGNN        # noqa: E402
from src.pipeline.schedule_bridge import (             # noqa: E402
    to_simulator_topology, decode_allgather, run_ground_truth,
)

# ============================================================
# Config
# ============================================================

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUT_PATH = ROOT / "data" / "raw" / "agent_solution_pool.pt"
FAILED_PATH = ROOT / "data" / "raw" / "agent_solution_pool_failed.pt"

SEED = 42
# IMPORTANT: harvest from the Agent's TRAIN pool only. These samples
# feed SimNet PRE-training; drawing them from agent val/test would let
# held-out topologies leak into the surrogate that later evaluates
# them.
AGENT_POOL_PATH = DATA_DIR / "pyg_agent_train.pt"

N_MEDIUM_PER_GRAPH = 2       # sampled-mask schedules per graph
N_LOW_PER_GRAPH = 2          # noised schedules per graph
LOW_ALPHA = 0.4              # low tier: alpha*p + (1-alpha)*U(0,1)

INCLUDE_ILP = False          # flip after implementing the converter
CANONICAL_PATH = DATA_DIR / "canonical_agent_train.pt"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Agent loading (same contract as everywhere else)
# ============================================================

def load_agent(device):
    ckpt = torch.load(MODEL_DIR / f"agent_best_seed{SEED}.pt",
                      map_location=device, weights_only=False)
    cfg = ckpt["config"]
    agent = StrongAgentGNN(
        node_in_dim=cfg["node_in_dim"], edge_in_dim=cfg["edge_in_dim"],
        global_in_dim=cfg["global_in_dim"],
        hidden_dim=cfg["hidden_dim"], heads=cfg["heads"],
        num_layers=cfg["num_layers"], dropout=cfg["dropout"],
        scheduling_activation=cfg["scheduling_activation"],
    ).to(device)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()
    for p in agent.parameters():
        p.requires_grad_(False)
    return agent, float(ckpt.get("tuned_threshold", 0.5))


@torch.no_grad()
def agent_probs(agent, data, device):
    u = data.u.unsqueeze(0) if data.u.dim() == 1 else data.u
    _, probs, sched = agent.predict(
        data.x.to(device), data.edge_index.to(device),
        data.edge_attr.to(device), batch=None, u=u.to(device),
    )
    return probs.cpu().numpy(), sched.cpu().numpy()


# ============================================================
# Quality tiers
# ============================================================

def tier_probs(probs, tier, rng):
    if tier == "agent_high":
        return probs                       # greedy at threshold
    if tier == "agent_medium":
        mask = (rng.random(len(probs)) < probs).astype(float)
        return np.clip(mask, 1e-6, 1 - 1e-6)
    if tier == "agent_low":
        noised = LOW_ALPHA * probs + (1 - LOW_ALPHA) * rng.random(
            len(probs))
        mask = (rng.random(len(probs)) < noised).astype(float)
        return np.clip(mask, 1e-6, 1 - 1e-6)
    raise ValueError(tier)


# ============================================================
# ILP-optimum hook (Step 2, first half of the data union)
# ============================================================

def ilp_schedule_to_policy(raw_schedule, data):
    """
    Convert the teacher's raw schedule events (preserved as
    `raw_schedule` in canonical samples) into a PolicyEntry list.
    The event format depends on solve_allgather_ilp — implement once
    the format is confirmed, then set INCLUDE_ILP = True.
    Expected mapping: each event (chunk, src->dst edge, epoch) becomes
    a single-hop PolicyEntry; same-chunk events chain as dependencies
    ordered by epoch.
    """
    raise NotImplementedError(
        "Implement using your ILP schedule event format."
    )


# ============================================================
# Main
# ============================================================

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    print("Device:", device)

    agent, threshold = load_agent(device)
    pool = torch.load(AGENT_POOL_PATH, map_location="cpu",
                      weights_only=False)
    print(f"Agent TRAIN pool: {len(pool)} graphs "
          f"(val/test excluded by design — see header)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    samples, failed = [], []
    rng = np.random.default_rng(SEED)
    sample_id = 0

    tiers = (
        [("agent_high", 1)]
        + [("agent_medium", N_MEDIUM_PER_GRAPH)]
        + [("agent_low", N_LOW_PER_GRAPH)]
    )

    for gi, data in enumerate(pool):
        try:
            probs, sched = agent_probs(agent, data, device)
            topo = to_simulator_topology(data)
        except Exception as e:
            failed.append({"pool_idx": gi, "stage": "agent/topo",
                           "error": str(e)})
            continue

        for tier, count in tiers:
            for rep in range(count):
                try:
                    p = tier_probs(probs, tier, rng)
                    policy, stats = decode_allgather(
                        data, p, sched, threshold
                    )
                    makespan = run_ground_truth(topo, policy)

                    samples.append({
                        # ---- generate_simnet_dataset.py schema ----
                        "sample_id": 100_000 + sample_id,
                        "seed": SEED,
                        "topology": topo,
                        "policy": policy,
                        "completion_time": makespan,
                        "tx_times": {},
                        "topology_type": str(data.topology_name),
                        "policy_type": tier,
                        "num_nodes": topo.number_of_nodes(),
                        "num_edges": topo.number_of_edges(),
                        "chunk_mb": float(data.message_size_bytes)
                        / (1024 ** 2),
                        "chunk_size_bytes":
                            float(data.message_size_bytes),
                        "num_policy_entries": len(policy),
                        # ---- provenance ----
                        "source_pool_idx": gi,
                        "quality_tier": tier,
                        "tier_rep": rep,
                        "decode_fallback_edges":
                            stats["fallback_edges"],
                    })
                    sample_id += 1
                except Exception as e:
                    failed.append({"pool_idx": gi, "tier": tier,
                                   "error": str(e)})

        if (gi + 1) % 10 == 0:
            torch.save(samples, OUT_PATH)
            torch.save(failed, FAILED_PATH)
            print(f"progress: {gi + 1}/{len(pool)} graphs, "
                  f"{len(samples)} samples")

    # ---- optional ILP half of the union ----
    if INCLUDE_ILP:
        canonical = torch.load(CANONICAL_PATH, map_location="cpu",
                               weights_only=False)
        for s in canonical:
            if "raw_schedule" not in s:
                continue
            try:
                # data-like shim for the bridge functions
                raise NotImplementedError  # see ilp_schedule_to_policy
            except NotImplementedError:
                print("ILP conversion not implemented — skipping.")
                break

    torch.save(samples, OUT_PATH)
    torch.save(failed, FAILED_PATH)

    print("=" * 70)
    print("DONE —", len(samples), "samples,", len(failed), "failed")
    print("tiers:", dict(Counter(s["quality_tier"]
                                 for s in samples)))
    times = np.array([s["completion_time"] for s in samples])
    for tier in ("agent_high", "agent_medium", "agent_low"):
        m = np.array([s["quality_tier"] == tier for s in samples])
        if m.any():
            print(f"  {tier}: median makespan "
                  f"{np.median(times[m]):.5g} s (n={m.sum()})")
    print("Saved:", OUT_PATH)
    print()
    print("NEXT STEP — merge into SimNet pre-training:")
    print("  in build_simnet_pyg.py, load this file alongside")
    print("  simnet_samples.pt and concatenate the raw lists before")
    print("  conversion. The size-extrapolation split and the")
    print("  policy-type breakdown in eval_simnet.py handle the new")
    print("  tiers automatically.")


if __name__ == "__main__":
    main()
