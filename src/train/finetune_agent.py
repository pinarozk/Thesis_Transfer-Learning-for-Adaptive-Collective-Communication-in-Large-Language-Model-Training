"""
Step 4 — Agent Fine-tuning with frozen SimNet (COCA).

Slide spec: freeze SimNet, update the Agent only;
Agent -> SimNet -> predicted reward -> Agent update; occasionally
validate with the real simulator.

Design decisions beyond the slide (thesis-documentable):

1. POLICY-GRADIENT THROUGH THE DISCRETE DECODER: the decode step
   (probabilities -> broadcast trees) is non-differentiable, so
   'SimNet as a differentiable teacher' cannot mean backprop through
   the schedule. We use REINFORCE with a self-normalized baseline:
   per graph, K edge-selection masks are sampled from Bernoulli(p),
   each is decoded and scored by SimNet, and the advantage of each
   sample against the K-sample mean drives the log-probability
   gradient. This is exact, unbiased, and honest about where the
   gradient comes from.

2. UNCERTAINTY-PENALIZED REWARD (anti-exploitation): a frozen
   surrogate WILL be exploited — the Agent drifts into regions where
   SimNet is wrong-and-optimistic. Reward = -(pred_log_makespan +
   LAMBDA_SIGMA * sigma): schedules that look fast only where SimNet
   is unsure are not rewarded. This is the concrete use of the
   heteroscedastic head and, together with #3, the answer to the
   surrogate-exploitation problem.

3. BC ANCHOR: a KL penalty to the frozen pre-fine-tuning policy keeps
   the Agent within the region where SimNet was calibrated in Step 3
   (trust region in spirit). The exploration/anchoring tradeoff
   (BETA_BC) is a natural ablation.

4. PERIODIC GROUND-TRUTH VALIDATION with an exploitation monitor:
   every VAL_EVERY steps the greedy policy is decoded and run in the
   REAL simulator on held-out graphs. We track surrogate-vs-real gap
   over training: if predicted reward keeps improving while real CCT
   does not, exploitation is detected and reported — not hidden.

Headline result: real-simulator CCT of the fine-tuned Agent vs the
imitation-only Agent, paired per held-out graph.
"""

import sys
import json
import math
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.agent_gnn import StrongAgentGNN        # noqa: E402
from src.models.simnet_gnn import SimNetGNN            # noqa: E402
from src.pipeline.schedule_bridge import (             # noqa: E402
    to_simulator_topology, decode_allgather,
    featurize_for_simnet, run_ground_truth,
    agent_per_source_chunk_bytes,
)

# ============================================================
# Paths / config
# ============================================================

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "src" / "eval" / "finetune"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TRAIN_POOL_PATH = DATA_DIR / "pyg_agent_train.pt"
VAL_POOL_PATH = DATA_DIR / "pyg_agent_val.pt"

NUM_STEPS = 600            # graph-level update steps
K_SAMPLES = 8              # sampled masks per graph (baseline group)
LR = 3e-5                  # small: we refine, not retrain
LAMBDA_SIGMA = 1.0         # uncertainty penalty weight (ablate)
BETA_BC = 0.5              # KL anchor to pre-finetune policy (ablate)
ENT_W = 1e-3               # mild exploration bonus
VAL_EVERY = 50             # ground-truth validation cadence
N_VAL_GRAPHS = 12          # simulator calls per validation
GRAD_CLIP = 1.0

RESULTS_PATH = OUT_DIR / f"finetune_seed{SEED}.json"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def inv(y):
    return float(np.power(10.0, y))


# ============================================================
# Loading
# ============================================================

def load_agent(device, trainable):
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
    if not trainable:
        agent.eval()
        for p in agent.parameters():
            p.requires_grad_(False)
    return agent, cfg, float(ckpt.get("tuned_threshold", 0.5))


def load_simnet(device):
    """Prefer the Step-3 calibrated checkpoint; fall back loudly."""
    cal = MODEL_DIR / f"simnet_calibrated_seed{SEED}.pt"
    base = MODEL_DIR / f"simnet_best_seed{SEED}.pt"
    path = cal if cal.exists() else base
    if path is base:
        print("WARNING: calibrated SimNet not found — using the "
              "uncalibrated checkpoint. Run calibrate_simnet.py "
              "first for the full COCA loop.")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = SimNetGNN(
        node_in_dim=cfg["node_in_dim"], edge_in_dim=cfg["edge_in_dim"],
        global_in_dim=cfg["global_in_dim"],
        hidden_dim=cfg["hidden_dim"], heads=cfg["heads"],
        num_layers=cfg["num_layers"], dropout=cfg["dropout"],
        predict_uncertainty=(cfg["loss_type"] == "nll"),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


# ============================================================
# Surrogate reward for one decoded schedule
# ============================================================

@torch.no_grad()
def surrogate_reward(simnet, sn_cfg, data, mask, sched_pred, device):
    """
    mask: 0/1 edge selection. Decodes (threshold irrelevant: the mask
    itself IS the selection, encoded as prob 1/0 with eps).
    Returns (reward, pred_log, sigma). reward = -(pred + lam*sigma).
    """
    pseudo_probs = np.clip(mask.astype(float), 1e-6, 1 - 1e-6)
    policy, stats = decode_allgather(data, pseudo_probs, sched_pred,
                                     threshold=0.5)
    topo = to_simulator_topology(data)
    g = featurize_for_simnet(topo, policy,
                             float(data.message_size_bytes))
    lb = (g.lb_log10.view(-1).to(device)
          if sn_cfg["target_mode"] == "residual" else None)
    pred, sigma = simnet.predict(
        g.x.to(device), g.edge_index.to(device),
        g.edge_attr.to(device), None, g.u.to(device), lb_log10=lb,
    )
    pred, sigma = float(pred.item()), float(sigma.item())
    reward = -(pred + LAMBDA_SIGMA * sigma)
    return reward, pred, sigma, stats


# ============================================================
# One REINFORCE step on one graph
# ============================================================

def rl_step(agent, ref_agent, simnet, sn_cfg, data, optimizer,
            device):
    x = data.x.to(device)
    ei = data.edge_index.to(device)
    ea = data.edge_attr.to(device)
    u = data.u
    if u.dim() == 1:
        u = u.unsqueeze(0)
    u = u.to(device)

    logits, sched_pred = agent(x, ei, ea, batch=None, u=u)
    probs = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    sched_np = sched_pred.detach().cpu().numpy()

    with torch.no_grad():
        ref_logits, _ = ref_agent(x, ei, ea, batch=None, u=u)
        ref_probs = torch.sigmoid(ref_logits).clamp(1e-4, 1 - 1e-4)

    # ---- sample K masks, score with the frozen surrogate ----
    masks, rewards, preds, sigmas = [], [], [], []
    for _ in range(K_SAMPLES):
        m = torch.bernoulli(probs).detach()
        r, p, s, _ = surrogate_reward(
            simnet, sn_cfg, data, m.cpu().numpy(), sched_np, device
        )
        masks.append(m)
        rewards.append(r); preds.append(p); sigmas.append(s)

    rewards_t = torch.tensor(rewards, device=device)
    baseline = rewards_t.mean()                # self-normalized
    adv = rewards_t - baseline
    if adv.std() > 1e-8:
        adv = adv / adv.std()

    # ---- REINFORCE loss ----
    pg = 0.0
    for m, a in zip(masks, adv):
        logp = (m * torch.log(probs)
                + (1 - m) * torch.log(1 - probs)).sum()
        pg = pg - a * logp
    pg = pg / K_SAMPLES

    # ---- BC anchor: KL(ref || current) per edge (contribution #3)
    kl = (ref_probs * torch.log(ref_probs / probs)
          + (1 - ref_probs) * torch.log((1 - ref_probs)
                                        / (1 - probs))).mean()

    # ---- entropy bonus ----
    ent = -(probs * torch.log(probs)
            + (1 - probs) * torch.log(1 - probs)).mean()

    loss = pg + BETA_BC * kl - ENT_W * ent

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in agent.parameters() if p.requires_grad], GRAD_CLIP
    )
    optimizer.step()

    return {
        "loss": float(loss), "pg": float(pg), "kl": float(kl),
        "mean_reward": float(rewards_t.mean()),
        "mean_pred_log": float(np.mean(preds)),
        "mean_sigma": float(np.mean(sigmas)),
    }


# ============================================================
# Ground-truth validation (contribution #4)
# ============================================================

@torch.no_grad()
def gt_validate(agent, simnet, sn_cfg, val_graphs, device,
                threshold):
    """Greedy decode -> real simulator + surrogate, per graph."""
    rows = []
    for data in val_graphs:
        u = data.u.unsqueeze(0) if data.u.dim() == 1 else data.u
        _, probs, sched = agent.predict(
            data.x.to(device), data.edge_index.to(device),
            data.edge_attr.to(device), batch=None,
            u=u.to(device), threshold=threshold,
        )
        probs = probs.cpu().numpy()
        sched = sched.cpu().numpy()
        policy, _ = decode_allgather(data, probs, sched, threshold)
        topo = to_simulator_topology(data)
        real = run_ground_truth(topo, policy)

        g = featurize_for_simnet(topo, policy,
                                 agent_per_source_chunk_bytes(data))
        lb = (g.lb_log10.view(-1).to(device)
              if sn_cfg["target_mode"] == "residual" else None)
        pred, _ = simnet.predict(
            g.x.to(device), g.edge_index.to(device),
            g.edge_attr.to(device), None, g.u.to(device),
            lb_log10=lb,
        )
        rows.append({"topology": str(data.topology_name),
                     "real": real,
                     "surrogate": inv(float(pred.item()))})
    return rows


# ============================================================
# Plots
# ============================================================

def plot_training(history):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].plot(history["step"], history["mean_reward"], lw=1.6,
                 color="#2C7FB8")
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Surrogate reward")
    axes[0].set_title("Surrogate reward during fine-tuning")
    axes[0].grid(alpha=0.3)

    if history["val_step"]:
        axes[1].plot(history["val_step"], history["val_real"],
                     "o-", lw=2, color="#54A24B",
                     label="Real simulator CCT")
        axes[1].plot(history["val_step"], history["val_surr"],
                     "s--", lw=1.6, color="#E45756",
                     label="Surrogate CCT")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Mean CCT on validation graphs (s)")
        axes[1].set_yscale("log")
        axes[1].set_title("Exploitation monitor: surrogate vs real")
        axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "finetune_training.png", dpi=300)
    plt.close(fig)


def plot_headline(before_rows, after_rows):
    b = np.array([r["real"] for r in before_rows])
    a = np.array([r["real"] for r in after_rows])
    labels = [r["topology"] for r in before_rows]
    x = np.arange(len(b)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(8.0, 0.9 * len(b)), 5.2))
    ax.bar(x - w / 2, b, w, label="Imitation-only Agent",
           color="#BBBBBB", edgecolor="black")
    ax.bar(x + w / 2, a, w, label="Fine-tuned Agent",
           color="#2C7FB8", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Real-simulator CCT (s)")
    ax.set_yscale("log")
    imp = float(np.mean((b - a) / b))
    ax.set_title(f"Step 4 headline — real CCT before/after "
                 f"fine-tuning (mean improvement: {imp:+.1%})")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "finetune_headline.png", dpi=300)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    print("Device:", device)
    print(f"K={K_SAMPLES}, lambda_sigma={LAMBDA_SIGMA}, "
          f"beta_bc={BETA_BC}")

    agent, _, threshold = load_agent(device, trainable=True)
    ref_agent, _, _ = load_agent(device, trainable=False)
    simnet, sn_cfg = load_simnet(device)

    train_pool = torch.load(TRAIN_POOL_PATH, map_location="cpu",
                            weights_only=False)
    val_pool = torch.load(VAL_POOL_PATH, map_location="cpu",
                          weights_only=False)
    rng = random.Random(SEED)
    val_graphs = rng.sample(val_pool,
                            min(N_VAL_GRAPHS, len(val_pool)))
    print(f"Train pool: {len(train_pool)}, "
          f"GT-validation graphs: {len(val_graphs)}")

    optimizer = torch.optim.AdamW(
        [p for p in agent.parameters() if p.requires_grad], lr=LR
    )

    # ---- headline baseline: imitation-only real CCT ----
    print("\nBaseline ground-truth validation (imitation-only) ...")
    before_rows = gt_validate(ref_agent, simnet, sn_cfg, val_graphs,
                              device, threshold)
    print(f"  mean real CCT: "
          f"{np.mean([r['real'] for r in before_rows]):.5g} s")

    history = defaultdict(list)
    agent.train()

    for step in range(1, NUM_STEPS + 1):
        data = rng.choice(train_pool)
        try:
            info = rl_step(agent, ref_agent, simnet, sn_cfg, data,
                           optimizer, device)
        except Exception as e:
            print(f"  step {step} skipped "
                  f"({data.topology_name}): {e}")
            continue

        history["step"].append(step)
        for k, v in info.items():
            history[k].append(v)

        if step % 20 == 0:
            print(f"step {step:04d} | reward "
                  f"{info['mean_reward']:.4f} | kl {info['kl']:.4f} "
                  f"| sigma {info['mean_sigma']:.4f}")

        if step % VAL_EVERY == 0:
            rows = gt_validate(agent, simnet, sn_cfg, val_graphs,
                               device, threshold)
            real = float(np.mean([r["real"] for r in rows]))
            surr = float(np.mean([r["surrogate"] for r in rows]))
            history["val_step"].append(step)
            history["val_real"].append(real)
            history["val_surr"].append(surr)
            print(f"  [GT val @ {step}] real {real:.5g} s | "
                  f"surrogate {surr:.5g} s | "
                  f"gap {(abs(surr - real) / real):.1%}")

    # ---- headline: after fine-tuning ----
    after_rows = gt_validate(agent, simnet, sn_cfg, val_graphs,
                             device, threshold)
    b = np.array([r["real"] for r in before_rows])
    a = np.array([r["real"] for r in after_rows])
    improvement = float(np.mean((b - a) / b))

    plot_training(dict(history))
    plot_headline(before_rows, after_rows)

    torch.save({
        "model_state_dict": agent.state_dict(),
        "seed": SEED,
        "finetune_config": {
            "num_steps": NUM_STEPS, "k_samples": K_SAMPLES,
            "lr": LR, "lambda_sigma": LAMBDA_SIGMA,
            "beta_bc": BETA_BC, "ent_w": ENT_W,
        },
    }, MODEL_DIR / f"agent_finetuned_seed{SEED}.pt")

    result = {
        "seed": SEED,
        "mean_real_cct_before": float(b.mean()),
        "mean_real_cct_after": float(a.mean()),
        "mean_relative_improvement": improvement,
        "per_graph": [
            {"topology": br["topology"],
             "before": br["real"], "after": ar["real"]}
            for br, ar in zip(before_rows, after_rows)
        ],
        "final_surrogate_real_gap": (
            abs(history["val_surr"][-1] - history["val_real"][-1])
            / history["val_real"][-1]
            if history["val_step"] else None
        ),
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=float)

    print("\n" + "=" * 70)
    print(f"HEADLINE — mean real CCT: {b.mean():.5g} s -> "
          f"{a.mean():.5g} s ({improvement:+.1%})")
    print("Saved:", RESULTS_PATH)
    print("Checkpoint:", MODEL_DIR / f"agent_finetuned_seed{SEED}.pt")


if __name__ == "__main__":
    main()
