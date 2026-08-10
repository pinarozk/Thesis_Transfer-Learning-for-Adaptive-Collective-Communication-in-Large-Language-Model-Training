"""
Agent GNN for neural CCL synthesis (routing + scheduling per edge).

Fixes vs. previous version (each is thesis-documentable)
--------------------------------------------------------
1. EDGE EMBEDDING BOTTLENECK REMOVED: edges were projected to only
   edge_in_dim (~7 dims) and never updated across layers, even though
   edge attributes (capacity, latency, tx cost) carry the primary signal
   in this problem AND the prediction targets live on edges. Edges are
   now embedded into hidden_dim and UPDATED at every layer from their
   endpoint node states (message-passing on edges, GN-block style).
2. JUMPING KNOWLEDGE: the edge readout now sees node/edge states from
   all layers (concatenated), not just the last one. Multi-hop and
   local information both reach the heads — helps when the useful
   receptive field varies with topology size (size generalization).
3. FAIL-LOUD INPUTS: nan_to_num silently buried upstream data bugs.
   Non-finite inputs now raise immediately with a diagnostic message.
   (The data pipeline already guarantees finiteness; the model should
   verify contracts, not repair violations.)
4. SAFE predict(): the old version permanently switched the module to
   eval mode as a side effect. Training mode is now saved and restored.
5. MASKED LOSS BUILT IN: compute_losses() implements exactly the
   training contract of the new dataset — Huber for load (edge chunk
   count/amount, log1p-scaled) on ALL edges, MSE for scheduling on USED
   edges only (sched_mask). Keeping it inside the model file makes the
   contract impossible to get wrong in a training script.
6. [REMOVED] post-hoc BCE calibration (temperature scaling): replaced
   by fix #8 below. A calibrated routing PROBABILITY has no meaning
   once routing is a regression target.
7. CONFIGURABLE DEPTH + weight init, minor cleanups.
8. ROUTING HEAD -> LOAD REGRESSION HEAD: real data showed
   routing_positive_ratio ~0.99-1.00 across all three teachers on
   these topologies (small/dense graphs + AllGather touch nearly
   every edge at least once) -- the binary "used or not" target is
   near-degenerate (a trivial always-positive baseline scores ~1.0
   F1). Per-edge chunk load is not degenerate (bottleneck edges carry
   far more traffic) and load>0 exactly where the old routing target
   was 1, so one regression head replaces the old binary head with no
   information loss and no separate mask.

I/O contract (matches convert_to_pyg.py)
-----------------------------------------
    x:          [num_nodes, node_in_dim]      (incl. RWSE)
    edge_index: [2, num_edges]                (directed; WAN asymmetry
                                               is representable because
                                               each direction is its own
                                               edge with its own attrs)
    edge_attr:  [num_edges, edge_in_dim]
    batch:      [num_nodes] (optional)
    u:          [num_graphs, global_in_dim] (optional)

    forward -> load_pred [num_edges] (log1p edge-load regression),
               scheduling_pred [num_edges]
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
)


# ============================================================
# Building blocks
# ============================================================

def mlp(in_dim, out_dim, dropout=0.0):
    layers = [
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.ReLU(),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class EdgeUpdate(nn.Module):
    """
    GN-block style edge update:
        e' = LN(e + MLP([h_src, h_dst, e]))
    Gives edges their own residual stream so edge states can accumulate
    multi-hop context instead of remaining frozen input features.
    """

    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, e, edge_index):
        src, dst = edge_index
        upd = self.mlp(torch.cat([h[src], h[dst], e], dim=-1))
        return self.norm(e + upd)


class AgentLayer(nn.Module):
    """One message-passing layer: node update (GATv2) + edge update."""

    def __init__(self, hidden_dim, heads, dropout):
        super().__init__()
        self.conv = GATv2Conv(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
            concat=True,
            edge_dim=hidden_dim,      # edges now live in hidden space
            dropout=dropout,
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_update = EdgeUpdate(hidden_dim, dropout)
        self.dropout = dropout

    def forward(self, h, e, edge_index):
        h_new = self.conv(h, edge_index, e)
        h_new = self.node_norm(h + h_new)
        h_new = F.relu(h_new)
        h_new = F.dropout(h_new, p=self.dropout, training=self.training)

        e_new = self.edge_update(h_new, e, edge_index)
        return h_new, e_new


# ============================================================
# Agent
# ============================================================

class StrongAgentGNN(nn.Module):

    def __init__(
        self,
        node_in_dim,
        edge_in_dim,
        global_in_dim=0,
        hidden_dim=96,
        heads=4,
        num_layers=3,
        dropout=0.15,
        use_graph_context=True,
        scheduling_activation="sigmoid",  # target lives in (0, 1]
    ):
        super().__init__()

        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads.")
        if scheduling_activation not in {"sigmoid", "none", "softplus"}:
            raise ValueError(
                "scheduling_activation must be 'sigmoid', 'none' or "
                "'softplus'."
            )
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")

        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.global_in_dim = global_in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_graph_context = use_graph_context
        self.scheduling_activation = scheduling_activation

        # ---- input encoders (both into hidden_dim; fix #1) ----
        self.node_proj = mlp(node_in_dim, hidden_dim)
        self.edge_proj = mlp(edge_in_dim, hidden_dim)

        # ---- message-passing stack ----
        self.layers = nn.ModuleList(
            AgentLayer(hidden_dim, heads, dropout)
            for _ in range(num_layers)
        )

        # ---- graph context (mean+max+sum over node states) ----
        if use_graph_context:
            self.graph_context_proj = mlp(hidden_dim * 3, hidden_dim)
            graph_context_dim = hidden_dim
        else:
            self.graph_context_proj = None
            graph_context_dim = 0

        # ---- external global features ----
        if global_in_dim and global_in_dim > 0:
            self.global_proj = mlp(global_in_dim, hidden_dim)
            global_context_dim = hidden_dim
        else:
            self.global_proj = None
            global_context_dim = 0

        # ---- edge readout with jumping knowledge (fix #2) ----
        # node states: (num_layers + 1) versions (input + each layer),
        # taken at both endpoints; edge states likewise.
        jk = num_layers + 1
        edge_repr_dim = (
            hidden_dim * 2 * jk        # src & dst node states, all layers
            + hidden_dim * jk          # edge states, all layers
            + graph_context_dim
            + global_context_dim
        )

        self.shared_edge_mlp = nn.Sequential(
            nn.Linear(edge_repr_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # load_head replaces the old binary routing_head (fix #8, see
        # module docstring): routing_positive_ratio is ~0.99-1.00
        # across all three teachers on these topologies (AllGather
        # uses nearly every edge at least once), so "used or not" is a
        # near-degenerate classification target. Per-edge chunk-load
        # (log1p-scaled) is not degenerate -- bottleneck edges carry
        # far more traffic -- and load>0 exactly where the old routing
        # target was 1, so this head subsumes it: no separate binary
        # head, no temperature calibration (there is no probability to
        # calibrate for a regression output).
        self.load_head = nn.Linear(hidden_dim // 2, 1)
        self.scheduling_head = nn.Linear(hidden_dim // 2, 1)

        self.apply(self._init_weights)

    # ============================================================
    # Init / validation helpers
    # ============================================================

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _check_finite(tensor, name):
        # Fail loudly (fix #3): a NaN here is an upstream bug that must
        # surface now, not as a mysteriously flat loss curve later.
        if not torch.isfinite(tensor).all():
            bad = (~torch.isfinite(tensor)).sum().item()
            raise ValueError(
                f"Non-finite values in {name}: {bad} entries. "
                f"Fix the data pipeline; the model will not mask this."
            )

    def _make_batch(self, x, batch):
        if batch is None:
            return torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return batch

    @staticmethod
    def _num_graphs(batch):
        if batch.numel() == 0:
            return 1
        return int(batch.max().item()) + 1

    def _graph_context(self, h, batch):
        if not self.use_graph_context:
            return None
        g = torch.cat(
            [
                global_mean_pool(h, batch),
                global_max_pool(h, batch),
                global_add_pool(h, batch),
            ],
            dim=-1,
        )
        return self.graph_context_proj(g)

    def _global_context(self, u, num_graphs, device):
        if self.global_proj is None:
            return None
        if u is None:
            return torch.zeros(num_graphs, self.hidden_dim, device=device)

        u = u.to(device).float()
        if u.dim() == 1:
            if num_graphs > 1 and u.numel() % num_graphs == 0:
                u = u.view(num_graphs, -1)
            else:
                u = u.unsqueeze(0)

        self._check_finite(u, "u")

        if u.size(0) == 1 and num_graphs > 1:
            u = u.expand(num_graphs, -1)
        if u.size(0) != num_graphs:
            raise ValueError(
                f"u has {u.size(0)} rows for {num_graphs} graphs."
            )
        if u.size(-1) != self.global_in_dim:
            raise ValueError(
                f"u dim mismatch: expected {self.global_in_dim}, "
                f"got {u.size(-1)}."
            )
        return self.global_proj(u)

    # ============================================================
    # Encoder
    # ============================================================

    def encode(self, x, edge_index, edge_attr):
        """Returns lists of node/edge states from every depth (JK)."""
        x = x.float()
        edge_attr = edge_attr.float()
        self._check_finite(x, "x")
        self._check_finite(edge_attr, "edge_attr")

        h = self.node_proj(x)
        e = self.edge_proj(edge_attr)

        h_states, e_states = [h], [e]
        for layer in self.layers:
            h, e = layer(h, e, edge_index)
            h_states.append(h)
            e_states.append(e)

        return h_states, e_states

    # ============================================================
    # Forward
    # ============================================================

    def forward(self, x, edge_index, edge_attr, batch=None, u=None):
        batch = self._make_batch(x, batch)
        num_graphs = self._num_graphs(batch)

        h_states, e_states = self.encode(x, edge_index, edge_attr)

        src, dst = edge_index
        edge_parts = []
        for h in h_states:                       # JK over node states
            edge_parts.append(h[src])
            edge_parts.append(h[dst])
        edge_parts.extend(e_states)              # JK over edge states

        h_last = h_states[-1]

        graph_context = self._graph_context(h_last, batch)
        if graph_context is not None:
            edge_parts.append(graph_context[batch[src]])

        global_context = self._global_context(u, num_graphs, x.device)
        if global_context is not None:
            edge_parts.append(global_context[batch[src]])

        edge_hidden = self.shared_edge_mlp(torch.cat(edge_parts, dim=-1))

        load_pred = self.load_head(edge_hidden).squeeze(-1)
        scheduling_raw = self.scheduling_head(edge_hidden).squeeze(-1)

        if self.scheduling_activation == "sigmoid":
            scheduling_pred = torch.sigmoid(scheduling_raw)
        elif self.scheduling_activation == "softplus":
            scheduling_pred = F.softplus(scheduling_raw)
        else:
            scheduling_pred = scheduling_raw

        return load_pred, scheduling_pred

    # ============================================================
    # Loss (fix #5) — the single source of truth for training
    # ============================================================

    def compute_losses(
        self,
        load_pred,
        scheduling_pred,
        y_load,
        y_scheduling,
        sched_mask,
        scheduling_weight=1.0,
        load_edge_weight=None,
    ):
        """
        load:       Huber (smooth L1) over ALL edges against log1p(chunk
                    count/amount). Unlike the old binary routing target,
                    0 is a valid, meaningful value here (unused edge),
                    so no mask and no pos_weight -- the class-imbalance
                    problem doesn't exist for a regression target.
                    Huber (not plain MSE) because load is heavy-tailed
                    (bottleneck edges can carry 10-100x the median),
                    and a few outlier edges must not dominate the loss.
        scheduling: MSE over USED edges only. An unused edge's "time" is
                    undefined; letting it contribute gradient corrupts
                    the head (the exact failure mode of the old target).

        load_edge_weight: optional per-edge weight tensor (same shape as
                    load_pred), e.g. upweighting switch-adjacent edges.
                    Added after the downstream CCT-gap measurement showed
                    switch-heavy topologies (DGX2_2_chassis) cost 33x vs
                    2.7x on switchless ones, traced to switch edges being
                    <10% of train edges vs 80% of that held-out test
                    topology's edges -- a train-time fix worth trying
                    even though it can't inject genuinely unseen switch
                    topology structure. None reproduces the old unweighted
                    behavior exactly.
        """
        if load_edge_weight is None:
            load_loss = F.smooth_l1_loss(load_pred, y_load)
        else:
            per_edge = F.smooth_l1_loss(load_pred, y_load, reduction="none")
            w = load_edge_weight.float()
            load_loss = (per_edge * w).sum() / w.sum().clamp(min=1e-8)

        mask = sched_mask.float()
        denom = mask.sum().clamp(min=1.0)
        scheduling_loss = (
            ((scheduling_pred - y_scheduling) ** 2) * mask
        ).sum() / denom

        total = load_loss + scheduling_weight * scheduling_loss
        return {
            "total": total,
            "load": load_loss,
            "scheduling": scheduling_loss,
        }

    # ============================================================
    # Prediction (fix #4: no permanent mode switch)
    # ============================================================
    # NOTE: the old post-hoc temperature-calibration hook (fix #6) is
    # gone -- it calibrated a BCE probability, and load_pred is a
    # regression output with no probability to calibrate. Downstream
    # code that consumed (routing_pred, routing_prob) -- notably
    # schedule_bridge.py's decode_allgather(), which used -log(prob)
    # as a Dijkstra edge weight -- needs to switch to load_pred
    # directly as an edge-preference score (higher predicted load =
    # more preferred edge; e.g. cost = -load_pred, or any monotonic
    # transform of it). Flagged, not silently left broken: that
    # rewrite is a separate, deliberate follow-up.

    @torch.no_grad()
    def predict(self, x, edge_index, edge_attr, batch=None, u=None,
                load_threshold=None):
        """
        load_threshold: cut in log1p(load) space for the derived
        is_used_pred convenience output. Default log1p(1) == the model
        predicting at least one whole chunk crossed the edge.
        """
        if load_threshold is None:
            load_threshold = math.log1p(1.0)
        was_training = self.training
        self.eval()
        try:
            load_pred, scheduling_pred = self.forward(
                x=x, edge_index=edge_index, edge_attr=edge_attr,
                batch=batch, u=u,
            )
            is_used_pred = (load_pred > load_threshold).float()
            return load_pred, is_used_pred, scheduling_pred
        finally:
            self.train(was_training)


# ============================================================
# Factory
# ============================================================

def build_agent_from_sample(
    sample,
    hidden_dim=96,
    heads=4,
    num_layers=3,
    dropout=0.15,
    use_graph_context=True,
    scheduling_activation="sigmoid",
):
    """Infer feature dims from a PyG Data sample and build the agent."""
    if not hasattr(sample, "x"):
        raise AttributeError("Sample has no node features: sample.x")
    if not hasattr(sample, "edge_attr"):
        raise AttributeError("Sample has no edge features: sample.edge_attr")

    node_in_dim = sample.x.size(-1)
    edge_in_dim = sample.edge_attr.size(-1)

    u = getattr(sample, "u", None)
    if u is not None:
        global_in_dim = u.size(0) if u.dim() == 1 else u.size(-1)
    else:
        global_in_dim = 0

    return StrongAgentGNN(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        global_in_dim=global_in_dim,
        hidden_dim=hidden_dim,
        heads=heads,
        num_layers=num_layers,
        dropout=dropout,
        use_graph_context=use_graph_context,
        scheduling_activation=scheduling_activation,
    )
