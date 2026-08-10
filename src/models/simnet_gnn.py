"""
SimNet: learned performance model (neural surrogate) for CCL schedules.
Maps (topology, schedule) graphs -> log10(makespan).

Fixes vs. previous version (thesis-documentable)
------------------------------------------------
1. EDGE READOUT ADDED (the big one): makespan is governed by the
   BOTTLENECK EDGE — structurally a max over per-edge busy times plus
   contention corrections. The old readout pooled only NODE states, so
   the quantity that determines the target was never read out directly.
   Edges are now embedded into hidden_dim, updated every layer
   (GN-block style), and pooled (mean+max+sum) alongside node states.
   The max-pool over edge states is the architectural analogue of the
   analytic bound the target follows.
2. EDGE EMBEDDING BOTTLENECK REMOVED: same fix as the Agent — edge
   features carry the physics (log serialization time, rates, trigger
   times) and were compressed to edge_in_dim and frozen across layers.
3. HETEROSCEDASTIC UNCERTAINTY HEAD: the COCA loop's calibration step
   (query the real simulator when the surrogate is unreliable, retrain
   if gap > tau) requires the surrogate to KNOW when it is unsure.
   The model now outputs (mean, log_var) and trains with Gaussian NLL;
   predicted sigma is the trigger signal for simulator queries and an
   OOD flag during Agent fine-tuning (guards against the Agent
   exploiting surrogate errors).
4. RESIDUAL-OVER-LOWER-BOUND TARGET MODE: the converter ships an
   analytic lower bound per graph. Predicting the residual
   y - log10(LB) instead of y makes the learning problem a bounded
   correction on top of exact physics — markedly better for size
   extrapolation. Supported natively via compute_loss / predict.
5. JUMPING KNOWLEDGE, configurable depth, Xavier init, fail-loud input
   checks, safe u reshaping — same conventions as the Agent model.

I/O contract (matches build_simnet_pyg.py)
------------------------------------------
    x [N, node_in], edge_index [2, E], edge_attr [E, edge_in],
    batch [N], u [num_graphs, global_in]
    forward -> (mean [num_graphs], log_var [num_graphs])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
)


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
    """e' = LN(e + MLP([h_src, h_dst, e])) — edges get their own
    residual stream (fix #2)."""

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
        return self.norm(
            e + self.mlp(torch.cat([h[src], h[dst], e], dim=-1))
        )


class SimNetLayer(nn.Module):
    def __init__(self, hidden_dim, heads, dropout):
        super().__init__()
        self.conv = GATv2Conv(
            hidden_dim, hidden_dim // heads, heads=heads,
            concat=True, edge_dim=hidden_dim, dropout=dropout,
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_update = EdgeUpdate(hidden_dim, dropout)
        self.dropout = dropout

    def forward(self, h, e, edge_index):
        h_new = self.conv(h, edge_index, e)
        h_new = self.node_norm(h + h_new)
        h_new = F.relu(h_new)
        h_new = F.dropout(h_new, p=self.dropout,
                          training=self.training)
        e_new = self.edge_update(h_new, e, edge_index)
        return h_new, e_new


class SimNetGNN(nn.Module):

    def __init__(
        self,
        node_in_dim,
        edge_in_dim,
        global_in_dim,
        hidden_dim=128,
        heads=4,
        num_layers=3,
        dropout=0.10,
        predict_uncertainty=True,
    ):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads.")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")

        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.global_in_dim = global_in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.predict_uncertainty = predict_uncertainty

        self.node_proj = mlp(node_in_dim, hidden_dim)
        self.edge_proj = mlp(edge_in_dim, hidden_dim)   # fix #2
        self.global_proj = mlp(global_in_dim, hidden_dim)

        self.layers = nn.ModuleList(
            SimNetLayer(hidden_dim, heads, dropout)
            for _ in range(num_layers)
        )

        # readout: node pools + EDGE pools (fix #1), with JK (fix #5)
        jk = num_layers + 1
        readout_dim = (
            hidden_dim * 3 * jk     # node mean/max/sum per depth
            + hidden_dim * 3 * jk   # edge mean/max/sum per depth
            + hidden_dim            # projected global features
        )

        self.regressor = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mean_head = nn.Linear(hidden_dim // 2, 1)
        self.logvar_head = (nn.Linear(hidden_dim // 2, 1)
                            if predict_uncertainty else None)

        self.apply(self._init_weights)

    # --------------------------------------------------------

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def _check_finite(t, name):
        if not torch.isfinite(t).all():
            raise ValueError(
                f"Non-finite values in {name} — fix the data "
                f"pipeline; the model will not mask this."
            )

    @staticmethod
    def _prepare_u(u, num_graphs, global_in_dim, device):
        if u is None:
            raise ValueError("SimNet requires global features u.")
        u = u.to(device).float()
        if u.dim() == 1:
            if num_graphs > 1 and u.numel() % num_graphs == 0:
                u = u.view(num_graphs, -1)
            else:
                u = u.unsqueeze(0)
        if u.size(0) != num_graphs:
            raise ValueError(
                f"u has {u.size(0)} rows for {num_graphs} graphs."
            )
        if u.size(-1) != global_in_dim:
            raise ValueError(
                f"u dim mismatch: expected {global_in_dim}, "
                f"got {u.size(-1)}."
            )
        return u

    # --------------------------------------------------------

    def forward(self, x, edge_index, edge_attr, batch, u):
        x = x.float()
        edge_attr = edge_attr.float()
        self._check_finite(x, "x")
        self._check_finite(edge_attr, "edge_attr")

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long,
                                device=x.device)
        num_graphs = int(batch.max().item()) + 1 \
            if batch.numel() else 1

        h = self.node_proj(x)
        e = self.edge_proj(edge_attr)

        h_states, e_states = [h], [e]
        for layer in self.layers:
            h, e = layer(h, e, edge_index)
            h_states.append(h)
            e_states.append(e)

        src = edge_index[0]
        edge_batch = batch[src]

        pools = []
        for hs in h_states:
            pools += [
                global_mean_pool(hs, batch),
                global_max_pool(hs, batch),
                global_add_pool(hs, batch),
            ]
        for es in e_states:                       # fix #1
            pools += [
                global_mean_pool(es, edge_batch),
                global_max_pool(es, edge_batch),
                global_add_pool(es, edge_batch),
            ]

        u = self._prepare_u(u, num_graphs, self.global_in_dim,
                            x.device)
        pools.append(self.global_proj(u))

        z = self.regressor(torch.cat(pools, dim=-1))
        mean = self.mean_head(z).squeeze(-1)

        if self.logvar_head is not None:
            log_var = self.logvar_head(z).squeeze(-1)
            log_var = log_var.clamp(-10.0, 4.0)   # numeric safety
        else:
            log_var = torch.zeros_like(mean)

        return mean, log_var

    # --------------------------------------------------------
    # Loss contract (fixes #3, #4)
    # --------------------------------------------------------

    def compute_loss(self, mean, log_var, y,
                     lb_log10=None, loss_type="nll"):
        """
        y:        log10(makespan), shape [num_graphs]
        lb_log10: optional log10(analytic lower bound); if given,
                  the model learns the RESIDUAL y - lb (fix #4) and
                  predict() adds the bound back.
        loss_type: 'nll' (heteroscedastic Gaussian) or 'mse'.
        """
        target = y - lb_log10 if lb_log10 is not None else y

        if loss_type == "mse" or self.logvar_head is None:
            return F.mse_loss(mean, target)
        if loss_type == "nll":
            # 0.5 * [ log_var + (err^2 / var) ], constant dropped
            inv_var = torch.exp(-log_var)
            return (0.5 * (log_var
                           + (mean - target) ** 2 * inv_var)).mean()
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    @torch.no_grad()
    def predict(self, x, edge_index, edge_attr, batch, u,
                lb_log10=None):
        """
        Returns (log10_makespan_pred, sigma). If lb_log10 is provided,
        the model output is treated as a residual and the bound is
        added back — must match how compute_loss was called in
        training. sigma is the predicted std in log10 space: the
        surrogate's own reliability signal for the COCA calibration
        loop (query the real simulator when sigma is high).
        """
        was_training = self.training
        self.eval()
        try:
            mean, log_var = self.forward(x, edge_index, edge_attr,
                                         batch, u)
            pred = mean + lb_log10 if lb_log10 is not None else mean
            sigma = torch.exp(0.5 * log_var)
            return pred, sigma
        finally:
            self.train(was_training)


# ============================================================
# Factory
# ============================================================

def build_simnet_from_sample(
    sample,
    hidden_dim=128,
    heads=4,
    num_layers=3,
    dropout=0.10,
    predict_uncertainty=True,
):
    node_in_dim = sample.x.size(-1)
    edge_in_dim = sample.edge_attr.size(-1)
    u = sample.u
    global_in_dim = u.size(-1) if u.dim() > 1 else u.size(0)

    return SimNetGNN(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        global_in_dim=global_in_dim,
        hidden_dim=hidden_dim,
        heads=heads,
        num_layers=num_layers,
        dropout=dropout,
        predict_uncertainty=predict_uncertainty,
    )
