"""
build_merged_dataset.py — RECONSTRUCTED.

The original module this name refers to (imported by
generate_teacher_dataset.py: BASE_DIR, create_gurobi_env, the 5
existing topology builders, solve_allgather_ilp) does not exist
anywhere in this repo, its git history, or the original newforthesis/
handoff folder -- confirmed by search, not assumed. Only its OUTPUT
(data/processed/allgather_teacher_dataset_incremental.pt) survived.

SCOPE OF THIS RECONSTRUCTION: solve_allgather_ilp is rebuilt here as a
general, parametric function extracted from
external/ilp_ccl_tree/ilp_model_NDv2_split.py's procedural script (tree
generation via tree_generate.py, MILP variables/constraints, schedule
extraction via results_process.py) -- that logic is topology-agnostic
given (adjacency, capacity, transfer_delay), so it generalizes cleanly.

NOT reconstructed: the 5 ORIGINAL topology builders (build_original_
8gpu_topology, build_original_plus_extra_topology, build_two_bridge_
topology, build_ring_cluster_topology, build_soft_leaf_spine_topology).
Their exact adjacency/capacity/latency values are not recoverable from
the canonical dataset alone (only build_original_8gpu_topology's
physics could be partially inferred from
external/ilp_ccl_tree/topo_generate.py's generate_delay_and_capacity,
which is itself hardcoded to that one 8-node family, not general).
generate_teacher_dataset.py therefore still cannot regenerate/extend
those 5 existing families until someone supplies or rewrites those 5
functions -- a separate, lower-priority gap that does NOT block new
switched-family generation (this module + switched_topologies.py are
self-contained and don't need the old builders).

Switch-awareness (the actual point of this reconstruction): unlike the
generic generate_all_gather_uniform_chunks used by the original script
(fine for the 5 existing switchless families), solve_allgather_ilp
here uses generate_all_gather_single_chunk_with_switch with EXPLICIT
exclude_nodes=switch_indices whenever switch_indices is non-empty --
never the module's own auto-detection heuristic (it assumes exactly
one switch with in/out-degree 2 on an odd node count, which does not
hold for the multi-switch dual_star/two_tier/switched_clusters
families switched_topologies.py generates).
"""

import math
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
from gurobipy import GRB, Model, LinExpr, quicksum

BASE_DIR = Path(__file__).resolve().parents[2]
ILP_DIR = BASE_DIR / "external" / "ilp_ccl_tree"
sys.path.append(str(ILP_DIR))
sys.path.append(str(BASE_DIR))

from network_parameter import (                              # noqa: E402
    generate_all_gather_uniform_chunks,
    generate_all_gather_single_chunk_with_switch,
)
from tree_generate import generate_and_select_trees            # noqa: E402
from results_process import extract_transmission_schedule      # noqa: E402

from src.data.teacher_common import events_to_targets           # noqa: E402


# ============================================================
# Gurobi environment
# ============================================================

def create_gurobi_env(output_flag=0):
    """Bare default env -- academic license picked up automatically
    from GRB_LICENSE_FILE / the standard install location."""
    import gurobipy as gp
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", output_flag)
    env.start()
    return env


# ============================================================
# General ILP solve (topology-agnostic; extracted + generalized from
# ilp_model_NDv2_split.py)
# ============================================================

def solve_allgather_ilp(
    topo_demo,               # N x N 0/1 adjacency (list of lists)
    topology_name,
    message_size,            # bytes (total AllGather message size)
    seed,
    gurobi_env,
    capacity,                # N x N absolute bandwidth matrix (same
                              # unit convention as switched_topologies.
                              # py / the original 50e9-scale constants)
    transfer_delay,           # N x N propagation-delay-class matrix:
                              # 0 on the diagonal, a small constant
                              # (e.g. 0.7e-6-1.6e-6) on real edges,
                              # np.inf where there is no edge -- same
                              # convention as generate_delay_and_capacity
    switch_indices=None,      # nodes excluded as AllGather source/dest
    source_indices=None,      # explicit GPU/source node list (defaults
                              # to all-nodes-minus-switch_indices)
    K=80,
    tree_number=8,
    fraction_split=4,
    time_limit_sec=300,
    enumeration_limit=2000,
    sample_tries=500,
    verbose=False,
):
    """
    Returns a sample dict matching the schema canonicalize_sample()
    expects for source_domain == "ilp_teacher":
        topology_name, message_size, node_feat, edge_index, edge_attr,
        routing_target, load_target, scheduling_target,
        completion_epoch, completion_time, schedule, objective,
        mip_gap, solver_status, switch_indices, source_indices, source.

    Targets are MATRICES (N x N), built via teacher_common.
    events_to_targets on the flattened per-(source,chunk) schedule --
    the exact same function TE-CCL/SyCCL use, since
    extract_transmission_schedule already emits events in the
    'src'=chunk-origin / 'from'-'to'=real edge / 'amount'=fractional
    convention events_to_targets expects for this teacher.
    """
    N = len(topo_demo)
    switch_indices = sorted(int(x) for x in (switch_indices or []))
    if source_indices is None:
        source_indices = sorted(set(range(N)) - set(switch_indices))
    else:
        source_indices = sorted(int(x) for x in source_indices)

    rng = np.random.default_rng(seed)

    cap = np.asarray(capacity, dtype=float)
    td = np.asarray(transfer_delay, dtype=float)

    fastest_linkrate = float(cap[np.isfinite(cap) & (cap > 0)].max())
    # BUG (found via teacher_replay.py cross-check, 2026-08-01): this
    # used to divide by len(source_indices), treating message_size as
    # a TOTAL demand split evenly across sources. Every other teacher
    # (TE-CCL, SyCCL, the original 5 ILP families) uses message_size
    # as each source's OWN full chunk size -- replaying a solved
    # switched-family schedule at the (correct, undivided) chunk size
    # showed teacher_sim_cct inflated ~90x+ vs solver-reported
    # completion_time on a 16-GPU family, far beyond the ~16x a naive
    # per-source split would predict once ceil()'d epoch-count effects
    # compound it. All 126 switched_topologies.py samples solved
    # before this fix used the wrong (smaller) chunk_size internally,
    # so their own completion_time/schedule are self-consistent but
    # NOT comparable to other teachers' at face value -- flagged as a
    # known issue pending a full re-solve, not silently patched here.
    chunk_size = float(message_size)
    epoch_duration = chunk_size / fastest_linkrate

    # capacity in chunks/epoch (>=1 for the fastest link by construction)
    with np.errstate(divide="ignore"):
        cap_chunks_per_epoch = cap * epoch_duration
    # delay in whole epochs (ceil), inf where unreachable
    delay_epochs = np.where(
        np.isfinite(td), np.ceil(td / epoch_duration), np.inf
    )

    chunk_number = 1
    s_range = source_indices
    n_range = range(0, tree_number * fraction_split)
    i_range = range(0, N)
    j_range = range(0, N)
    c_range = range(0, chunk_number)
    k_range = range(0, K)
    d_range = source_indices

    # ---- trees (generic node-agnostic tree search over full adjacency) ----
    _, selected_trees = generate_and_select_trees(
        topo_demo, tree_number, enumeration_limit=enumeration_limit,
        sample_tries=sample_tries, verbose=verbose, do_plot=False,
    )
    copy_number = fraction_split
    total_tree_number = tree_number * copy_number
    selected_trees_copied = {}
    for (s, idx), edges in selected_trees.items():
        for copy_idx in range(copy_number):
            new_idx = idx * copy_number + copy_idx
            if new_idx < total_tree_number:
                selected_trees_copied[(s, new_idx)] = edges.copy()
    tree_lib = selected_trees_copied

    T = {}
    for s in s_range:
        for n in n_range:
            for j in j_range:
                for i in i_range:
                    T[j, i, s, n] = 0
    for (s, n), edge_list in tree_lib.items():
        for (i, j) in edge_list:
            T[i, j, s, n] = 1

    # ---- demand (switch-aware) ----
    if switch_indices:
        D, _ = generate_all_gather_single_chunk_with_switch(
            topo_demo, include_self=True, exclude_nodes=switch_indices,
            chunk_number=chunk_number,
        )
    else:
        D, _ = generate_all_gather_uniform_chunks(
            topo_demo, chunk_number, include_self=True,
        )

    model = Model("ILP_Model", env=gurobi_env)
    model.Params.TimeLimit = time_limit_sec
    model.Params.Seed = int(seed) % 2_000_000_000

    r = np.zeros((N, len(n_range), N, N, chunk_number, K)).tolist()
    for s, n, i, j, c, k in product(s_range, n_range, i_range, j_range,
                                    c_range, k_range):
        r[s][n][i][j][c][k] = model.addVar(
            lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
            name=f"route_frac_{s}_{n}_{i}_{j}_{c}_{k}")

    y = np.zeros((N, len(n_range), chunk_number)).tolist()
    for s, n, c in product(s_range, n_range, c_range):
        y[s][n][c] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                  name=f"tree_frac_s{s}_n{n}_c{c}")

    B = np.zeros((N, N, chunk_number, K)).tolist()
    for s, i, c, k in product(s_range, i_range, c_range, k_range):
        B[s][i][c][k] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                                     name=f"buffer_{s}_{i}_{c}_{k}")

    B_hold = np.zeros((N, N, chunk_number, K)).tolist()
    for s, i, c, k in product(s_range, i_range, c_range, k_range):
        B_hold[s][i][c][k] = model.addVar(
            lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
            name=f"buffer_hold_{s}_{i}_{c}_{k}")

    R = np.zeros((N, N, chunk_number, K)).tolist()
    for s, d, c, k in product(s_range, d_range, c_range, k_range):
        R[s][d][c][k] = model.addVar(
            lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
            name=f"demand_satisfied_{s}_{d}_{c}_{k}")

    z = np.zeros((N, len(n_range), N, N, chunk_number, K)).tolist()
    for s, n, i, j, c, k in product(s_range, n_range, i_range, j_range,
                                    c_range, k_range):
        z[s][n][i][j][c][k] = model.addVar(
            vtype=GRB.BINARY,
            name=f"send_ind_s{s}_n{n}_i{i}_j{j}_c{c}_k{k}")

    # init buffers
    for s, i, c in product(s_range, i_range, c_range):
        model.addConstr(B[s][i][c][0] == (D[s, i, c] if i == s else 0.0))
        model.addConstr(B_hold[s][i][c][0] == (D[s, i, c] if i == s else 0.0))

    for s, c, i, k in product(s_range, c_range, i_range, range(1, K)):
        if i == s:
            continue
        summation = quicksum(
            r[s][n][j][i][c][k - int(delay_epochs[j, i])]
            for j, n in product(j_range, n_range)
            if T[j, i, s, n] == 1
            and np.isfinite(delay_epochs[j, i])
            and (k - int(delay_epochs[j, i])) >= 0
        )
        model.addConstr(B[s][i][c][k - 1] + summation == B[s][i][c][k])

    for s, c, i, k in product(s_range, c_range, i_range, range(1, K)):
        if i == s:
            continue
        summation = quicksum(
            r[s][n][j][i][c][k - int(delay_epochs[j, i])]
            for j, n in product(j_range, n_range)
            if T[j, i, s, n] == 1
            and np.isfinite(delay_epochs[j, i])
            and (k - int(delay_epochs[j, i])) >= 0
        )
        model.addConstr(B_hold[s][i][c][k - 1] + summation == B_hold[s][i][c][k])

    for s, n, i, j, c, k in product(s_range, n_range, i_range, j_range,
                                    c_range, k_range):
        if T.get((i, j, s, n), 0) == 1:
            model.addConstr(r[s][n][i][j][c][k] <= B_hold[s][i][c][k])

    for s, d, c, k in product(s_range, d_range, c_range, range(K - 1)):
        if D[s][d][c] > 0:
            model.addConstr(R[s][d][c][k] == B_hold[s][d][c][k])
        else:
            model.addConstr(R[s][d][c][k] == 0.0)

    k_last = K - 1
    for s, d, c in product(s_range, d_range, c_range):
        terms = []
        for n, j in product(n_range, j_range):
            if T.get((j, d, s, n), 0) != 1:
                continue
            if not np.isfinite(delay_epochs[j, d]):
                continue
            t_idx = k_last - int(delay_epochs[j, d])
            if t_idx < 0:
                continue
            terms.append(r[s][n][j][d][c][t_idx])
        incoming_sum = quicksum(terms) if terms else 0.0
        model.addConstr(R[s][d][c][k_last] == B_hold[s][d][c][k_last] + incoming_sum)

    for s, d, c in product(s_range, d_range, c_range):
        model.addConstr(R[s][d][c][K - 1] == D[s, d, c])

    for s, c in product(s_range, c_range):
        model.addConstr(quicksum(y[s][n][c] for n in n_range) == 1.0)
    for s, n, c in product(s_range, n_range, c_range):
        model.addConstr(y[s][n][c] <= 1.0 / copy_number)

    for s, n, i, j, c, k in product(s_range, n_range, i_range, j_range,
                                    c_range, k_range):
        if T.get((i, j, s, n), 0) == 1:
            model.addConstr(r[s][n][i][j][c][k] <= y[s][n][c])
            model.addConstr(r[s][n][i][j][c][k] <= z[s][n][i][j][c][k])
            model.addConstr(
                r[s][n][i][j][c][k] >= y[s][n][c] - (1 - z[s][n][i][j][c][k]))
        else:
            model.addConstr(r[s][n][i][j][c][k] == 0.0)
            model.addConstr(z[s][n][i][j][c][k] == 0)

    for s, n, i, j, c in product(s_range, n_range, i_range, j_range, c_range):
        if T.get((i, j, s, n), 0) == 1:
            model.addConstr(
                quicksum(r[s][n][i][j][c][k] for k in k_range) == y[s][n][c])

    for i, j, k in product(i_range, j_range, k_range):
        cap_ij = float(cap_chunks_per_epoch[i, j])
        if cap_ij <= 0:
            continue
        beta_here = int(np.ceil(1.0 / cap_ij))
        expr = LinExpr(0.0)
        for l in range(beta_here):
            if (k - l) < 0:
                continue
            for s, n, c in product(s_range, n_range, c_range):
                expr.add(r[s][n][i][j][c][k - l])
        model.addConstr(expr <= 1)

    objective_opt = LinExpr(0.0)
    for s, d, c, k in product(s_range, d_range, c_range, k_range):
        if D[s, d, c] > 0:
            objective_opt.add(1.0 / (k + 1) * R[s][d][c][k], -1)
    model.setObjective(objective_opt, GRB.MINIMIZE)

    t0 = time.time()
    model.optimize()
    solve_time = time.time() - t0

    solver_status = str(model.Status)
    mip_gap = None
    completion_epoch = K - 1
    schedule_events = []
    objective_val = None

    if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) \
            and model.SolCount > 0:
        mip_gap = float(model.MIPGap)
        objective_val = float(model.ObjVal)

        raw_schedule = extract_transmission_schedule(
            r, s_range, c_range, n_range, K)
        for (s, c), entries in raw_schedule.items():
            for e in entries:
                schedule_events.append({
                    "src": e["src"], "from": e["from"], "to": e["to"],
                    "epoch": e["epoch"], "amount": e["amount"],
                })

        # completion epoch: latest epoch any r > eps fires (a tighter,
        # honest bound than the K-1 fallback; matches how TE-CCL/SyCCL
        # derive completion_epoch from their own event epochs).
        if schedule_events:
            completion_epoch = max(e["epoch"] for e in schedule_events)
    else:
        raise RuntimeError(
            f"solve_allgather_ilp: no usable solution for {topology_name} "
            f"(status={solver_status}, SolCount={model.SolCount})"
        )

    routing_target, load_target, scheduling_target, _ = events_to_targets(
        schedule_events, N)

    completion_time = (completion_epoch + 1) * epoch_duration

    # STORAGE-ONLY rescale: schedule_bridge.py's CAPACITY_TO_BPS=8e9
    # converts canonical edge_attr's stored capacity value (GB/s-scale,
    # matching the existing ilp_teacher/ring_cluster convention where
    # recover_physics() finds raw~10.7 there) back into bits/sec via
    # capacity * 8e9. Our own `cap` here is already an absolute bps
    # value (switched_topologies.py's capacity_bps, e.g. 50e9/ratio),
    # used correctly above for the SOLVER's own epoch-duration math in
    # real seconds -- but storing it AS-IS would make CAPACITY_TO_BPS
    # multiply by a further 8e9, producing an astronomically large
    # simulated link rate downstream (near-zero, message-size-
    # independent decoded CCT -- exactly the bug this comment fixes,
    # caught via cct_gap.py showing agent_cct constant at ~4us
    # regardless of message_size on the new switched-family holdouts).
    # Dividing by 8e9 here is an exact round-trip: recover_physics()
    # x CAPACITY_TO_BPS reconstructs precisely this function's own
    # intended bps value, decoupled from the solver's internal units.
    CAPACITY_STORAGE_SCALE = 8e9

    return {
        "topology_name": topology_name,
        "message_size": float(message_size),
        "node_feat": torch.zeros(N, 1),
        "edge_index": torch.tensor(
            [[i, j] for i in range(N) for j in range(N)
             if i != j and cap[i, j] > 0], dtype=torch.long).t().contiguous(),
        "edge_attr": torch.tensor(
            [[cap[i, j] / CAPACITY_STORAGE_SCALE, td[i, j]]
             for i in range(N) for j in range(N)
             if i != j and cap[i, j] > 0], dtype=torch.float),
        "routing_target": routing_target,
        "load_target": load_target,
        "scheduling_target": scheduling_target,
        "completion_epoch": int(completion_epoch),
        "completion_time": float(completion_time),
        "schedule": schedule_events,
        "objective": objective_val,
        "mip_gap": mip_gap,
        "solver_status": solver_status,
        "switch_indices": switch_indices,
        "source_indices": source_indices,
        "num_nodes": N,
        "source": "ilp_teacher",
        "collective": "AllGather",
    }
