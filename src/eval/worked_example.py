"""
worked_example.py — pull ONE real test instance through the whole
pipeline and print slide-ready numbers.

Purpose: the supervisor asked for a concrete walk-through — this
topology, these feature values, this prediction, this decoded
schedule, this simulated time, versus the teacher. Everything printed
here comes from the actual checkpoints and datasets; nothing is
illustrative.

Pick a SMALL instance so the numbers fit on a slide. ring_cluster
(8 GPUs, switchless, ILP teacher) is the natural choice: few edges,
a clean teacher schedule, and it sits in the test split, so it is
genuinely unseen by the model.

Usage:
    python worked_example.py                       # default pick
    python worked_example.py --topology ring_cluster --size 4e6
    python worked_example.py --list                # what is available
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "CCL_Simulator"))

from src.models.agent_gnn import StrongAgentGNN                 # noqa
from src.pipeline.schedule_bridge import (                      # noqa
    to_simulator_topology, decode_allgather, run_ground_truth,
    featurize_for_simnet, agent_per_source_chunk_bytes,
)
from src.pipeline.teacher_replay import teacher_sim_cct          # noqa

DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
SEED = 42


def load_agent(device):
    ck = torch.load(MODELS / f"agent_best_seed{SEED}.pt",
                    map_location=device, weights_only=False)
    cfg = ck["config"]
    m = StrongAgentGNN(
        node_in_dim=cfg["node_in_dim"], edge_in_dim=cfg["edge_in_dim"],
        global_in_dim=cfg["global_in_dim"], hidden_dim=cfg["hidden_dim"],
        heads=cfg["heads"], num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        scheduling_activation=cfg["scheduling_activation"],
    ).to(device)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m, cfg


def try_load_simnet(device):
    """Optional: calibrated surrogate, if the checkpoint exists."""
    for name in (f"simnet_calibrated_seed{SEED}.pt",
                 f"simnet_best_seed{SEED}.pt", "simnet_best.pt"):
        p = MODELS / name
        if not p.exists():
            continue
        try:
            from src.models.simnet_gnn import SimNetGNN
            ck = torch.load(p, map_location=device, weights_only=False)
            if "config" not in ck:
                print(f"  [simnet] {name} has no config — skipped")
                continue
            cfg = ck["config"]
            m = SimNetGNN(
                node_in_dim=cfg["node_in_dim"],
                edge_in_dim=cfg["edge_in_dim"],
                global_in_dim=cfg["global_in_dim"],
                hidden_dim=cfg["hidden_dim"], heads=cfg["heads"],
                num_layers=cfg["num_layers"], dropout=cfg["dropout"],
                predict_uncertainty=(cfg["loss_type"] == "nll"),
            ).to(device)
            m.load_state_dict(ck["model_state_dict"])
            m.eval()
            print(f"  [simnet] using {name}")
            return m, cfg
        except Exception as e:                       # noqa: BLE001
            print(f"  [simnet] {name} failed to load: {e}")
    return None, None


def fmt(v, n=4):
    return f"{v:.{n}g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default="ring_cluster")
    ap.add_argument("--size", type=float, default=None,
                    help="message_size_bytes; default = median available")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu")
    pyg = torch.load(DATA / "pyg_agent_test.pt", map_location="cpu",
                     weights_only=False)
    canon = torch.load(DATA / "canonical_agent_test.pt",
                       map_location="cpu", weights_only=False)

    if args.list:
        from collections import Counter
        c = Counter((str(d.topology_name),
                     float(d.message_size_bytes)) for d in pyg)
        for (t, m), n in sorted(c.items()):
            print(f"{t:32s} {m:>12.0f} B   x{n}")
        return

    cand = [d for d in pyg if str(d.topology_name) == args.topology]
    if not cand:
        raise SystemExit(f"No test graphs named {args.topology}. "
                         f"Run with --list.")
    sizes = sorted({float(d.message_size_bytes) for d in cand})
    target = args.size if args.size else sizes[len(sizes) // 2]
    data = min(cand, key=lambda d: abs(float(d.message_size_bytes)
                                       - target))
    ms = float(data.message_size_bytes)

    # matching canonical record (same topology, same size, first match)
    raw = next((s for s in canon
                if str(s["topology_name"]) == args.topology
                and abs(float(s["message_size_bytes"]) - ms) < 1e-6),
               None)

    N = int(data.x.size(0))
    E = int(data.edge_index.size(1))
    node_names = list(data.node_feature_names)
    edge_names = list(data.edge_feature_names)

    print("=" * 68)
    print(f"WORKED EXAMPLE — {args.topology} @ {ms:,.0f} B")
    print("=" * 68)

    # ---------- 1. the instance ----------
    print("\n[1] INSTANCE")
    print(f"  nodes N = {N}   directed edges E = {E}")
    src_i = node_names.index("is_source")
    n_src = int((data.x[:, src_i] > 0.5).sum())
    print(f"  sources (is_source=1) = {n_src}"
          f"   relays = {N - n_src}")
    print(f"  per-source chunk = "
          f"{agent_per_source_chunk_bytes(data):,.0f} B"
          f"   (message_size = {ms:,.0f} B)")
    if raw is not None:
        print(f"  teacher = {raw.get('source_domain')}   "
              f"mip_gap = {raw.get('mip_gap')}   "
              f"events = {len(raw.get('raw_schedule', []))}")

    # ---------- 2. features of ONE node and ONE edge ----------
    print("\n[2] FEATURE VALUES (one node, one edge — slide material)")
    v = 0
    print(f"  node {v}:")
    for name, val in zip(node_names, data.x[v].tolist()):
        print(f"      {name:16s} {fmt(val)}")
    # pick the busiest teacher edge so the example is meaningful
    y = data.y_load.view(-1)
    k = int(torch.argmax(y).item())
    u, w = data.edge_index[0, k].item(), data.edge_index[1, k].item()
    print(f"  edge {k}:  {u} -> {w}   (the most loaded edge in the "
          f"teacher schedule)")
    for name, val in zip(edge_names, data.edge_attr[k].tolist()):
        print(f"      {name:28s} {fmt(val)}")

    # ---------- 3. agent prediction ----------
    agent, _ = load_agent(device)
    u_vec = data.u if data.u.dim() == 2 else data.u.unsqueeze(0)
    with torch.no_grad():
        load_pred, sched_pred = agent(
            data.x, data.edge_index, data.edge_attr, None, u_vec)
    load_pred = load_pred.view(-1)
    sched_pred = sched_pred.view(-1)

    print("\n[3] AGENT PREDICTION")
    print(f"  on that edge:  y_true = {fmt(y[k].item())}"
          f"   y_pred = {fmt(load_pred[k].item())}"
          f"   (log1p scale)")
    print(f"                 chunks_true = "
          f"{fmt(np.expm1(y[k].item()), 3)}"
          f"   chunks_pred = "
          f"{fmt(np.expm1(load_pred[k].item()), 3)}")
    r = np.corrcoef(load_pred.numpy(), y.numpy())[0, 1]
    sp = __import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(
        load_pred.numpy(), y.numpy()).correlation
    mae = float(torch.mean(torch.abs(load_pred - y)))
    print(f"  whole graph:   Pearson r = {fmt(r,3)}   "
          f"Spearman = {fmt(sp,3)}   MAE = {fmt(mae,3)}")

    # ---------- 4. decode ----------
    policy, stats = decode_allgather(
        data, load_pred.numpy(), sched_pred.numpy())
    topo = to_simulator_topology(data)
    print("\n[4] DECODED SCHEDULE")
    print(f"  policy entries = {stats['policy_entries']}"
          f"   trees = {stats['trees_built']}"
          f"   fallback (repaired) edges = {stats['fallback_edges']}")
    print("  first 3 transfers:")
    for e in policy[:3]:
        dep = e.dependency[0] if e.dependency else "-"
        print(f"      {e.chunk_id:14s} {e.src} -> {e.dst}"
              f"   {e.chunk_size_bytes:,.0f} B   dep={dep}")

    # ---------- 5. simulate + compare ----------
    agent_cct = run_ground_truth(topo, policy)
    print("\n[5] SIMULATED COMPLETION TIME")
    print(f"  agent  CCT = {agent_cct:.6g} s")
    if raw is not None and raw.get("raw_schedule"):
        t_cct, t_stats = teacher_sim_cct(
            data, raw["raw_schedule"], ms)
        print(f"  teacher CCT = {t_cct:.6g} s"
              f"   (same simulator, same topology conversion)")
        print(f"  gap_sim = {agent_cct / t_cct:.3f}x"
              f"   missing_arrival = {t_stats['missing_arrival']}")
        if raw.get("completion_time"):
            print(f"  [reference] teacher's own reported time = "
                  f"{float(raw['completion_time']):.6g} s"
                  f"   -> replay_ratio = "
                  f"{t_cct / float(raw['completion_time']):.2f}x")

    # ---------- 6. simnet on the same schedule ----------
    simnet, scfg = try_load_simnet(device)
    if simnet is not None:
        g = featurize_for_simnet(
            topo, policy, agent_per_source_chunk_bytes(data))
        lb = (g.lb_log10.view(-1)
              if scfg.get("target_mode") == "residual" else None)
        with torch.no_grad():
            pred, sigma = simnet.predict(
                g.x, g.edge_index, g.edge_attr, None, g.u,
                lb_log10=lb)
        pred_s = float(10 ** pred.item())
        print("\n[6] SIMNET ON THE SAME DECODED SCHEDULE")
        print(f"  predicted CCT = {pred_s:.6g} s"
              f"   sigma = {float(sigma.item()):.3f} (log10)")
        print(f"  true (simulator) = {agent_cct:.6g} s"
              f"   APE = "
              f"{abs(pred_s - agent_cct) / agent_cct * 100:.1f}%")
        print(f"  analytic lower bound = "
              f"{float(g.analytic_lower_bound):.6g} s")
    else:
        print("\n[6] SIMNET — no loadable checkpoint found; skipped")

    print("\n" + "=" * 68)
    print("Copy the bracketed blocks straight onto the slide.")


if __name__ == "__main__":
    main()
