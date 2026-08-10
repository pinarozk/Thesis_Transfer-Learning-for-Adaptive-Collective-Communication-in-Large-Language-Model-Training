"""
Step 3 — SimNet Calibration (COCA).

Slide spec: freeze the Agent, feed its scheduling decisions to SimNet,
compare SimNet's predicted time with the real simulator, and locally
retrain SimNet if the gap exceeds tau (e.g. 5%). Goal: align SimNet
with the real simulator WITHIN THE AGENT'S EXPLORED REGION.

What this implementation adds beyond the slide (each is a thesis
contribution point):

1. SIGMA-GUIDED QUERYING (active calibration): real-simulator queries
   are the expensive resource. We rank candidate points by SimNet's
   own predicted sigma and show — with a lift curve against random
   ordering — that high-sigma points concentrate the large-gap cases.
   This turns the heuristic 'check the simulator sometimes' into a
   query-budget-efficient policy and empirically validates sigma as
   the trigger signal.

2. ANTI-FORGETTING REPLAY: naive local retraining on Agent-region
   points causes catastrophic forgetting of the original SimNet
   distribution. Each fine-tuning batch mixes REPLAY_RATIO of original
   training graphs. Before/after metrics are reported on BOTH the
   Agent region (should improve) and the original test set (should not
   degrade) — the honest two-sided evaluation.

3. FEASIBLE-BY-CONSTRUCTION EVALUATION POINTS: the Agent's raw output
   is decoded through schedule_bridge (with its repair fallback), so
   every calibration point is an executable schedule — exactly the
   distribution SimNet must be accurate on in Step 4.

Outputs: calibrated checkpoint simnet_calibrated_seed{S}.pt,
before/after JSON, gap histogram, lift curve, pred-vs-true overlay.
"""

import sys
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.agent_gnn import StrongAgentGNN        # noqa: E402
from src.models.simnet_gnn import SimNetGNN            # noqa: E402
from src.pipeline.schedule_bridge import (             # noqa: E402
    agent_to_schedule, featurize_for_simnet, run_ground_truth,
    agent_per_source_chunk_bytes,
)

# ============================================================
# Paths / config
# ============================================================

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "src" / "eval" / "calibration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AGENT_POOL_PATHS = [DATA_DIR / "pyg_agent_val.pt",
                    DATA_DIR / "pyg_agent_test.pt"]
SIMNET_TRAIN_PATH = DATA_DIR / "simnet_pyg_train.pt"
SIMNET_TEST_PATH = DATA_DIR / "simnet_pyg_test.pt"

SEED = 42                       # which agent/simnet seed pair to use
TAU = 0.05                      # slide's 5% gap threshold
QUERY_BUDGET = 120              # max real-simulator calls
REPLAY_RATIO = 0.5              # replay share in each fine-tune batch
FT_EPOCHS = 40
FT_LR = 1e-4
FT_PATIENCE = 8
HOLDOUT_FRAC = 0.2              # of collected points, for early stop

RESULTS_PATH = OUT_DIR / f"calibration_seed{SEED}.json"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def inv(y):
    return np.power(10.0, np.asarray(y, dtype=float))


# ============================================================
# Loading (strict contracts, same as elsewhere)
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


def load_simnet(device):
    ckpt = torch.load(MODEL_DIR / f"simnet_best_seed{SEED}.pt",
                      map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = SimNetGNN(
        node_in_dim=cfg["node_in_dim"], edge_in_dim=cfg["edge_in_dim"],
        global_in_dim=cfg["global_in_dim"],
        hidden_dim=cfg["hidden_dim"], heads=cfg["heads"],
        num_layers=cfg["num_layers"], dropout=cfg["dropout"],
        predict_uncertainty=(cfg["loss_type"] == "nll"),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, cfg


def attach_lb(dataset):
    for g in dataset:
        if not hasattr(g, "lb_log10"):
            lb = max(float(g.analytic_lower_bound), 1e-12)
            g.lb_log10 = torch.tensor([math.log10(lb)],
                                      dtype=torch.float)
    return dataset


# ============================================================
# Phase A — collect Agent-region calibration points
# ============================================================

@torch.no_grad()
def collect_points(agent, simnet, cfg, agent_pool, device,
                   threshold):
    """
    For each agent-pool graph: decode -> SimNet(pred, sigma).
    Points are then SORTED BY SIGMA (descending) and the top
    QUERY_BUDGET get real-simulator calls; a random ordering of the
    same size is also queried for the lift-curve comparison.
    """
    target_mode = cfg["target_mode"]
    candidates = []
    for i, data in enumerate(agent_pool):
        try:
            topo, policy, probs, sched, stats = agent_to_schedule(
                agent, data, device, threshold
            )
            g = featurize_for_simnet(topo, policy,
                                     agent_per_source_chunk_bytes(data))
            lb = (g.lb_log10.view(-1).to(device)
                  if target_mode == "residual" else None)
            pred, sigma = simnet.predict(
                g.x.to(device), g.edge_index.to(device),
                g.edge_attr.to(device), None, g.u.to(device),
                lb_log10=lb,
            )
            candidates.append({
                "pool_idx": i,
                "topology": str(data.topology_name),
                "g": g, "topo": topo, "policy": policy,
                "pred_log": float(pred.item()),
                "sigma": float(sigma.item()),
                "decode_stats": stats,
            })
        except Exception as e:
            print(f"  skip pool[{i}] ({data.topology_name}): {e}")

    print(f"Candidates decoded: {len(candidates)}")

    # sigma-ordered vs random-ordered querying (contribution #1)
    by_sigma = sorted(candidates, key=lambda c: -c["sigma"])
    rng = random.Random(SEED)
    by_random = list(candidates)
    rng.shuffle(by_random)

    budget = min(QUERY_BUDGET, len(candidates))
    query_set = {id(c) for c in by_sigma[:budget]}
    # add random-order points for the lift comparison (may overlap)
    query_set |= {id(c) for c in by_random[:budget]}

    queried = []
    for c in candidates:
        if id(c) not in query_set:
            continue
        true = run_ground_truth(c["topo"], c["policy"])
        c["true"] = true
        c["true_log"] = math.log10(true)
        c["ape"] = abs(inv(c["pred_log"]) - true) / true
        queried.append(c)

    return candidates, queried, by_sigma[:budget], by_random[:budget]


def lift_curve(order_a, order_b, tau, tag_a="sigma-ordered",
               tag_b="random"):
    """Fraction of large-gap points (ape > tau) found vs #queries."""
    def curve(order):
        found, xs, ys = 0, [], []
        total = sum(1 for c in order if c.get("ape", 0) > tau)
        for i, c in enumerate(order, 1):
            if c.get("ape", 0) > tau:
                found += 1
            xs.append(i)
            ys.append(found / max(1, total))
        return xs, ys

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for order, tag, color in [(order_a, tag_a, "#E45756"),
                              (order_b, tag_b, "#BBBBBB")]:
        order = [c for c in order if "ape" in c]
        xs, ys = curve(order)
        ax.plot(xs, ys, lw=2.2, label=tag, color=color)
    ax.set_xlabel("Number of real-simulator queries")
    ax.set_ylabel(f"Fraction of gap>{tau:.0%} points discovered")
    ax.set_title("Sigma-guided querying vs random "
                 "(query-budget efficiency)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "lift_curve.png", dpi=300)
    plt.close(fig)


# ============================================================
# Phase B — local retraining with anti-forgetting replay
# ============================================================

def finetune(simnet, cfg, queried, replay_pool, device):
    target_mode = cfg["target_mode"]
    loss_type = cfg["loss_type"]

    # calibration Data objects with truth attached
    cal = []
    for c in queried:
        g = c["g"]
        g.y = torch.tensor([c["true_log"]], dtype=torch.float)
        cal.append(g)
    rng = random.Random(SEED)
    rng.shuffle(cal)
    n_hold = max(1, int(HOLDOUT_FRAC * len(cal)))
    holdout, cal_train = cal[:n_hold], cal[n_hold:]

    def lb_arg(batch):
        return (batch.lb_log10.view(-1)
                if target_mode == "residual" else None)

    def eval_mape(graphs):
        apes = []
        for g in graphs:
            lb = (g.lb_log10.view(-1).to(device)
                  if target_mode == "residual" else None)
            pred, _ = simnet.predict(
                g.x.to(device), g.edge_index.to(device),
                g.edge_attr.to(device), None, g.u.to(device),
                lb_log10=lb,
            )
            apes.append(abs(inv(pred.item()) - inv(g.y.item()))
                        / inv(g.y.item()))
        return float(np.mean(apes))

    opt = torch.optim.AdamW(simnet.parameters(), lr=FT_LR,
                            weight_decay=1e-5)
    best = eval_mape(holdout)
    best_state = {k: v.clone() for k, v in
                  simnet.state_dict().items()}
    patience = 0
    print(f"  fine-tune start: holdout MAPE {best:.4f}")

    exclude = None
    for epoch in range(1, FT_EPOCHS + 1):
        # batch = calibration points + replay (contribution #2)
        n_replay = int(len(cal_train) * REPLAY_RATIO
                       / max(1e-9, 1 - REPLAY_RATIO))
        batch_graphs = cal_train + rng.sample(
            replay_pool, min(n_replay, len(replay_pool))
        )
        rng.shuffle(batch_graphs)
        if exclude is None:
            keys = set()
            for g in batch_graphs[:20]:
                keys.update(g.keys())
            exclude = sorted(keys - {"x", "edge_index", "edge_attr",
                                     "u", "y", "lb_log10"})
        loader = DataLoader(batch_graphs, batch_size=16,
                            shuffle=True, exclude_keys=exclude)

        simnet.train()
        for batch in loader:
            batch = batch.to(device)
            mean, log_var = simnet(batch.x, batch.edge_index,
                                   batch.edge_attr, batch.batch,
                                   batch.u)
            loss = simnet.compute_loss(mean, log_var,
                                       batch.y.view(-1),
                                       lb_log10=lb_arg(batch),
                                       loss_type=loss_type)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(simnet.parameters(), 2.0)
            opt.step()

        m = eval_mape(holdout)
        if m < best:
            best, patience = m, 0
            best_state = {k: v.clone() for k, v in
                          simnet.state_dict().items()}
        else:
            patience += 1
            if patience >= FT_PATIENCE:
                break
        if epoch % 5 == 0:
            print(f"  ft epoch {epoch:02d}: holdout MAPE {m:.4f} "
                  f"(best {best:.4f})")

    simnet.load_state_dict(best_state)
    return best


# ============================================================
# Two-sided evaluation helpers
# ============================================================

@torch.no_grad()
def region_mape(simnet, cfg, queried, device):
    target_mode = cfg["target_mode"]
    apes = []
    for c in queried:
        g = c["g"]
        lb = (g.lb_log10.view(-1).to(device)
              if target_mode == "residual" else None)
        pred, _ = simnet.predict(
            g.x.to(device), g.edge_index.to(device),
            g.edge_attr.to(device), None, g.u.to(device),
            lb_log10=lb,
        )
        apes.append(abs(inv(pred.item()) - c["true"]) / c["true"])
    return apes


@torch.no_grad()
def original_test_mape(simnet, cfg, test_graphs, device):
    target_mode = cfg["target_mode"]
    apes = []
    for g in test_graphs:
        lb = (g.lb_log10.view(-1).to(device)
              if target_mode == "residual" else None)
        pred, _ = simnet.predict(
            g.x.to(device), g.edge_index.to(device),
            g.edge_attr.to(device), None, g.u.to(device),
            lb_log10=lb,
        )
        apes.append(abs(inv(pred.item()) - inv(g.y.item()))
                    / inv(g.y.item()))
    return float(np.mean(apes))


def plot_gap_hist(before, after, tau):
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    bins = np.linspace(0, max(0.2, max(before + after)), 30)
    ax.hist(before, bins=bins, alpha=0.6, label="Before calibration",
            color="#BBBBBB", edgecolor="black", linewidth=0.3)
    ax.hist(after, bins=bins, alpha=0.6, label="After calibration",
            color="#2C7FB8", edgecolor="black", linewidth=0.3)
    ax.axvline(tau, ls="--", c="black", lw=1.4,
               label=f"tau = {tau:.0%}")
    ax.set_xlabel("SimNet APE on Agent-region schedules")
    ax.set_ylabel("Count")
    ax.set_title("Step 3 — Surrogate gap before/after calibration")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gap_before_after.png", dpi=300)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    print("Device:", device)

    agent, threshold = load_agent(device)
    simnet, cfg = load_simnet(device)

    agent_pool = []
    for p in AGENT_POOL_PATHS:
        agent_pool += torch.load(p, map_location="cpu",
                                 weights_only=False)
    print("Agent-region pool:", len(agent_pool))

    replay_pool = attach_lb(torch.load(SIMNET_TRAIN_PATH,
                                       map_location="cpu",
                                       weights_only=False))
    simnet_test = attach_lb(torch.load(SIMNET_TEST_PATH,
                                       map_location="cpu",
                                       weights_only=False))

    # ---- Phase A: collect + query ----
    candidates, queried, sigma_order, random_order = collect_points(
        agent, simnet, cfg, agent_pool, device, threshold
    )
    lift_curve(sigma_order, random_order, TAU)

    gaps_before = region_mape(simnet, cfg, queried, device)
    orig_before = original_test_mape(simnet, cfg, simnet_test, device)
    frac_over = float(np.mean(np.array(gaps_before) > TAU))
    print(f"\nBEFORE — Agent-region MAPE: "
          f"{np.mean(gaps_before):.4f} "
          f"({frac_over:.0%} of points exceed tau={TAU:.0%}); "
          f"original-test MAPE: {orig_before:.4f}")

    result = {
        "seed": SEED, "tau": TAU,
        "query_budget": QUERY_BUDGET,
        "n_candidates": len(candidates),
        "n_queried": len(queried),
        "decode_fallback_rate": float(np.mean(
            [c["decode_stats"]["fallback_edges"]
             / max(1, c["decode_stats"]["policy_entries"])
             for c in candidates])),
        "before": {
            "agent_region_mape": float(np.mean(gaps_before)),
            "frac_over_tau": frac_over,
            "original_test_mape": orig_before,
        },
        "retrained": False,
    }

    # ---- Phase B: retrain if the slide's condition holds ----
    if np.mean(gaps_before) > TAU:
        print(f"\nGap > tau — locally retraining SimNet "
              f"(replay ratio {REPLAY_RATIO}) ...")
        finetune(simnet, cfg, queried, replay_pool, device)

        gaps_after = region_mape(simnet, cfg, queried, device)
        orig_after = original_test_mape(simnet, cfg, simnet_test,
                                        device)
        print(f"AFTER  — Agent-region MAPE: "
              f"{np.mean(gaps_after):.4f}; "
              f"original-test MAPE: {orig_after:.4f} "
              f"(was {orig_before:.4f})")

        plot_gap_hist(gaps_before, gaps_after, TAU)

        result["retrained"] = True
        result["after"] = {
            "agent_region_mape": float(np.mean(gaps_after)),
            "frac_over_tau": float(np.mean(
                np.array(gaps_after) > TAU)),
            "original_test_mape": orig_after,
        }
        torch.save({
            "model_state_dict": simnet.state_dict(),
            "config": cfg,
            "calibration": result,
        }, MODEL_DIR / f"simnet_calibrated_seed{SEED}.pt")
        print("Saved calibrated checkpoint.")
    else:
        print(f"\nGap <= tau — no retraining needed; "
              f"the uncalibrated SimNet is already aligned.")
        plot_gap_hist(gaps_before, gaps_before, TAU)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=float)
    print("Saved:", RESULTS_PATH)


if __name__ == "__main__":
    main()
