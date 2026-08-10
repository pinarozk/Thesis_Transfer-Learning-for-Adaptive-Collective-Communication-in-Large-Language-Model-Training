import sys
import json
from pathlib import Path

import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.models.agent_gnn import StrongAgentGNN


DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

TEST_PYG = DATA_DIR / "pyg_agent_test.pt"
CKPT_PATH = MODEL_DIR / "agent_best.pt"

OUT_DIR = BASE_DIR / "src" / "eval" / "illustrative_examples" / "agent"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROUTING_THRESHOLD = 0.5
MIN_USED_RATIO = 0.05


def safe_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def edge_align_target(data, target_name):
    target = getattr(data, target_name).float()
    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    num_nodes = data.x.size(0)

    if target.dim() == 1 and target.numel() == num_edges:
        setattr(data, target_name, target.view(-1))
        return

    if target.dim() == 2 and target.size(0) == num_edges and target.size(1) == 1:
        setattr(data, target_name, target.view(-1))
        return

    if target.dim() == 2 and target.size(0) == num_nodes and target.size(1) == num_nodes:
        src, dst = edge_index
        setattr(data, target_name, target[src, dst].view(-1))
        return

    if target.dim() == 1 and target.numel() == num_nodes * num_nodes:
        mat = target.view(num_nodes, num_nodes)
        src, dst = edge_index
        setattr(data, target_name, mat[src, dst].view(-1))
        return

    raise ValueError(f"Cannot align {target_name}: {tuple(target.shape)}")


def standardize_sample(data, node_dim, edge_dim):
    edge_align_target(data, "y_routing")
    edge_align_target(data, "y_scheduling")

    if data.x.size(1) < node_dim:
        pad = torch.zeros(data.x.size(0), node_dim - data.x.size(1))
        data.x = torch.cat([data.x, pad], dim=1)
    elif data.x.size(1) > node_dim:
        data.x = data.x[:, :node_dim]

    if data.edge_attr.size(1) < edge_dim:
        pad = torch.zeros(data.edge_attr.size(0), edge_dim - data.edge_attr.size(1))
        data.edge_attr = torch.cat([data.edge_attr, pad], dim=1)
    elif data.edge_attr.size(1) > edge_dim:
        data.edge_attr = data.edge_attr[:, :edge_dim]

    return data


def prepare_u(data):
    if not hasattr(data, "u") or data.u is None:
        return None
    u = data.u
    if u.dim() == 1:
        u = u.unsqueeze(0)
    return u


def tensor_to_value(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        if x.numel() == 1:
            return x.item()
        return x.detach().cpu().tolist()
    return x


def get_meta(data):
    fields = [
        "topology_name",
        "topology",
        "topology_type",
        "collective",
        "collective_type",
        "mode",
        "message_size",
        "message_size_bytes",
        "chunk_size_bytes",
    ]

    meta = {}
    for f in fields:
        if hasattr(data, f):
            meta[f] = tensor_to_value(getattr(data, f))
    return meta


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ckpt = safe_load(CKPT_PATH)
    test_data = safe_load(TEST_PYG)

    # -------------------------------------------------------
    # FIND A MEANINGFUL ILLUSTRATIVE SAMPLE
    # -------------------------------------------------------

    selected_idx = None
    selected_data = None

    for idx, candidate in enumerate(test_data):

        candidate = standardize_sample(
            candidate,
            node_dim=ckpt["node_in_dim"],
            edge_dim=ckpt["edge_in_dim"],
        )

        used_ratio = candidate.y_routing.float().mean().item()

        if used_ratio >= MIN_USED_RATIO:
            selected_idx = idx
            selected_data = candidate
            break

    if selected_data is None:
        raise ValueError(
            "No illustrative sample found with non-zero routing targets."
        )

    EXAMPLE_IDX = selected_idx
    data = selected_data

    print(f"\nSelected illustrative example index: {EXAMPLE_IDX}")
    print(
        f"Routing target used-edge ratio: "
        f"{data.y_routing.float().mean().item():.4f}"
    )

    # -------------------------------------------------------
    # BUILD MODEL
    # -------------------------------------------------------

    model = StrongAgentGNN(
        node_in_dim=ckpt["node_in_dim"],
        edge_in_dim=ckpt["edge_in_dim"],
        global_in_dim=ckpt["global_in_dim"],
        hidden_dim=ckpt["hidden_dim"],
        heads=ckpt["heads"],
        dropout=ckpt["dropout"],
        use_graph_context=True,
        scheduling_activation=ckpt.get(
            "scheduling_activation",
            "sigmoid",
        ),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # -------------------------------------------------------
    # PREPARE INPUTS
    # -------------------------------------------------------

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device)

    u = prepare_u(data)
    if u is not None:
        u = u.to(device)

    batch = torch.zeros(
        x.size(0),
        dtype=torch.long,
        device=device,
    )

    # -------------------------------------------------------
    # FORWARD PASS
    # -------------------------------------------------------

    routing_logits, scheduling_pred = model(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch,
        u=u,
    )

    routing_prob = torch.sigmoid(routing_logits)
    routing_pred = (routing_prob >= ROUTING_THRESHOLD).float()

    y_routing = data.y_routing.float().view(-1).to(device)
    y_scheduling = data.y_scheduling.float().view(-1).to(device)

    # -------------------------------------------------------
    # BUILD EDGE TABLE
    # -------------------------------------------------------

    src = edge_index[0].detach().cpu().numpy()
    dst = edge_index[1].detach().cpu().numpy()

    rows = []

    for e in range(edge_index.size(1)):
        rows.append({
            "edge_id": e,
            "src": int(src[e]),
            "dst": int(dst[e]),

            "routing_target":
                float(y_routing[e].cpu()),

            "routing_prob":
                float(routing_prob[e].cpu()),

            "routing_pred":
                int(routing_pred[e].cpu().item()),

            "scheduling_target":
                float(y_scheduling[e].cpu()),

            "scheduling_pred":
                float(scheduling_pred[e].cpu()),

            "abs_scheduling_error":
                float(
                    abs(
                        scheduling_pred[e]
                        - y_scheduling[e]
                    ).cpu()
                ),
        })

    df = pd.DataFrame(rows)

    df.to_csv(
        OUT_DIR
        / f"agent_example_{EXAMPLE_IDX}_edge_predictions.csv",
        index=False,
    )

    # -------------------------------------------------------
    # METRICS
    # -------------------------------------------------------

    used = y_routing > 0.5

    routing_acc = (
        (routing_pred == y_routing)
        .float()
        .mean()
        .item()
    )

    if used.sum() > 0:
        sched_mae = (
            torch.mean(
                torch.abs(
                    scheduling_pred[used]
                    - y_scheduling[used]
                )
            ).item()
        )
    else:
        sched_mae = None

    summary = {
        "example_idx": EXAMPLE_IDX,
        "num_nodes": int(data.x.size(0)),
        "num_edges": int(data.edge_index.size(1)),
        "routing_accuracy": routing_acc,

        "routing_target_used_edge_ratio":
            float(y_routing.mean().cpu()),

        "routing_pred_used_edge_ratio":
            float(routing_pred.mean().cpu()),

        "scheduling_mae_used_edges_only":
            sched_mae,

        "metadata":
            get_meta(data),
    }

    with open(
        OUT_DIR
        / f"agent_example_{EXAMPLE_IDX}_summary.json",
        "w",
    ) as f:
        json.dump(summary, f, indent=2)

    # -------------------------------------------------------
    # ROUTING PLOT
    # -------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 4.5))

    x_pos = np.arange(len(df))

    ax.scatter(
        x_pos,
        df["routing_target"],
        label="Target",
        s=40,
    )

    ax.scatter(
        x_pos,
        df["routing_pred"],
        label="Predicted",
        s=18,
        alpha=0.80,
    )

    ax.set_title(
        f"Illustrative Example {EXAMPLE_IDX}: "
        f"Routing Prediction per Edge"
    )

    ax.set_xlabel("Directed Edge ID")
    ax.set_ylabel("Routing Label")

    ax.set_yticks([0, 1])

    ax.grid(True, alpha=0.25)

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUT_DIR
        / f"agent_example_{EXAMPLE_IDX}_routing_prediction.png",
        dpi=300,
    )

    plt.close(fig)

    # -------------------------------------------------------
    # SCHEDULING PLOT
    # -------------------------------------------------------

    used_df = df[
        df["routing_target"] > 0.5
    ].copy()

    if not used_df.empty:

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.scatter(
            used_df["scheduling_target"],
            used_df["scheduling_pred"],
            s=35,
            alpha=0.75,
        )

        min_v = min(
            used_df["scheduling_target"].min(),
            used_df["scheduling_pred"].min(),
        )

        max_v = max(
            used_df["scheduling_target"].max(),
            used_df["scheduling_pred"].max(),
        )

        ax.plot(
            [min_v, max_v],
            [min_v, max_v],
            linestyle="--",
            label="Perfect agreement",
        )

        ax.set_title(
            f"Illustrative Example {EXAMPLE_IDX}: "
            f"Scheduling Prediction"
        )

        ax.set_xlabel("Target Scheduling Score")
        ax.set_ylabel("Predicted Scheduling Score")

        ax.grid(True, alpha=0.25)

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            OUT_DIR
            / f"agent_example_{EXAMPLE_IDX}_scheduling_prediction.png",
            dpi=300,
        )

        plt.close(fig)

    # -------------------------------------------------------
    # SELECTED ROUTING EDGES
    # -------------------------------------------------------

    selected = df[
        df["routing_pred"] == 1
    ][[
        "edge_id",
        "src",
        "dst",
        "routing_prob",
        "scheduling_pred",
    ]]

    selected.to_csv(
        OUT_DIR
        / f"agent_example_{EXAMPLE_IDX}_selected_policy_edges.csv",
        index=False,
    )

    # -------------------------------------------------------
    # DONE
    # -------------------------------------------------------

    print("\nSaved Agent illustrative example outputs to:")
    print(OUT_DIR)

    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()