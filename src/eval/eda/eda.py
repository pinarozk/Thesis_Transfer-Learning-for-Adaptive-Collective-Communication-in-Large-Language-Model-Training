from pathlib import Path
from collections import Counter

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


BASE_DIR = Path(__file__).resolve().parents[3]

DATA_PATH = BASE_DIR / "data" / "processed" / "canonical_agent_dataset.pt"

OUT_DIR = BASE_DIR / "src" / "eval" / "eda_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


dataset = torch.load(DATA_PATH, weights_only=False)

print("=" * 80)
print("DATASET LOADED")
print("Samples:", len(dataset))


# ============================================================
# Aggregate statistics
# ============================================================

rows = []

for sample in dataset:

    routing_target = sample["routing_edge_target"]
    scheduling_target = sample["scheduling_edge_target"]

    used_edges = int((routing_target > 0).sum().item())

    rows.append({
        "source": sample["source_domain"],
        "topology": sample["topology_name"],
        "collective": sample["collective"],
        "mode": sample["mode"],

        "num_nodes": sample["num_nodes"],
        "num_edges": sample["num_edges"],

        "message_size_bytes": sample["message_size_bytes"],

        "routing_active_edges": used_edges,
        "routing_density": used_edges / sample["num_edges"],

        "avg_scheduling_target":
            float(scheduling_target[routing_target > 0].mean().item())
            if used_edges > 0 else 0.0,
    })

df = pd.DataFrame(rows)

print("\nDataset summary:")
print(df.head())


# ============================================================
# BASIC COUNTS
# ============================================================

print("\nTopology counts:")
print(df["topology"].value_counts())

print("\nSource counts:")
print(df["source"].value_counts())

print("\nCollective counts:")
print(df["collective"].value_counts())


# ============================================================
# Plot helper
# ============================================================

def save_plot(name):
    path = OUT_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print("Saved:", path)


# ============================================================
# 1. Node count distribution
# ============================================================

plt.figure(figsize=(7,5))

df["num_nodes"].hist(bins=10)

plt.title("Node Count Distribution")
plt.xlabel("Number of Nodes")
plt.ylabel("Frequency")

save_plot("node_count_distribution")


# ============================================================
# 2. Edge count distribution
# ============================================================

plt.figure(figsize=(7,5))

df["num_edges"].hist(bins=10)

plt.title("Edge Count Distribution")
plt.xlabel("Number of Edges")
plt.ylabel("Frequency")

save_plot("edge_count_distribution")


# ============================================================
# 3. Routing density distribution
# ============================================================

plt.figure(figsize=(7,5))

df["routing_density"].hist(bins=20)

plt.title("Routing Density Distribution")
plt.xlabel("Used Edge Ratio")
plt.ylabel("Frequency")

save_plot("routing_density_distribution")


# ============================================================
# 4. Message size vs active edges
# ============================================================

plt.figure(figsize=(7,5))

plt.scatter(
    np.log10(df["message_size_bytes"]),
    df["routing_active_edges"],
)

plt.title("Message Size vs Active Routing Edges")
plt.xlabel("log10(Message Size Bytes)")
plt.ylabel("Active Routing Edges")

save_plot("message_vs_active_edges")


# ============================================================
# 5. Topology vs routing density
# ============================================================

plt.figure(figsize=(9,5))

df.boxplot(column="routing_density", by="topology")

plt.title("Routing Density by Topology")
plt.suptitle("")
plt.xlabel("Topology")
plt.ylabel("Routing Density")

plt.xticks(rotation=20)

save_plot("routing_density_by_topology")


# ============================================================
# 6. Degree distribution across all graphs
# ============================================================

all_degrees = []

for sample in dataset:

    edge_index = sample["edge_index"]

    G = nx.DiGraph()

    for u, v in edge_index.t().tolist():
        G.add_edge(u, v)

    degrees = [d for _, d in G.degree()]
    all_degrees.extend(degrees)

plt.figure(figsize=(7,5))

plt.hist(all_degrees, bins=20)

plt.title("Global Degree Distribution")
plt.xlabel("Node Degree")
plt.ylabel("Frequency")

save_plot("global_degree_distribution")


# ============================================================
# 7. Graph density distribution
# ============================================================

graph_densities = []

for sample in dataset:

    n = sample["num_nodes"]
    e = sample["num_edges"]

    density = e / (n * (n - 1))

    graph_densities.append(density)

plt.figure(figsize=(7,5))

plt.hist(graph_densities, bins=20)

plt.title("Graph Density Distribution")
plt.xlabel("Density")
plt.ylabel("Frequency")

save_plot("graph_density_distribution")


# ============================================================
# 8. Routing target imbalance
# ============================================================

all_targets = []

for sample in dataset:
    all_targets.extend(
        sample["routing_edge_target"].tolist()
    )

all_targets = np.array(all_targets)

positive_ratio = (all_targets > 0).mean()

print("\nRouting positive ratio:", positive_ratio)

plt.figure(figsize=(5,5))

plt.bar(
    ["Unused Edge", "Used Edge"],
    [
        (all_targets == 0).sum(),
        (all_targets > 0).sum(),
    ]
)

plt.title("Routing Target Imbalance")

save_plot("routing_target_imbalance")


# ============================================================
# 9. Scheduling target distribution
# ============================================================

sched_values = []

for sample in dataset:

    routing = sample["routing_edge_target"]
    sched = sample["scheduling_edge_target"]

    used = sched[routing > 0]

    sched_values.extend(used.tolist())

plt.figure(figsize=(7,5))

plt.hist(sched_values, bins=20)

plt.title("Scheduling Target Distribution")
plt.xlabel("Normalized First Usage Epoch")
plt.ylabel("Frequency")

save_plot("scheduling_target_distribution")


# ============================================================
# 10. Switch-edge utilization analysis
# ============================================================

switch_usage = []
non_switch_usage = []

for sample in dataset:

    edge_attr = sample["edge_attr"]
    routing = sample["routing_edge_target"]

    switch_flag = edge_attr[:, 2]

    for s, r in zip(switch_flag.tolist(), routing.tolist()):

        if s > 0:
            switch_usage.append(r)
        else:
            non_switch_usage.append(r)

plt.figure(figsize=(6,5))

plt.bar(
    ["Non-switch edges", "Switch edges"],
    [
        np.mean(non_switch_usage),
        np.mean(switch_usage),
    ]
)

plt.ylabel("Mean Routing Usage")
plt.title("Switch vs Non-switch Edge Usage")

save_plot("switch_vs_non_switch_usage")


# ============================================================
# 11. Correlation matrix
# ============================================================

corr_cols = [
    "num_nodes",
    "num_edges",
    "message_size_bytes",
    "routing_active_edges",
    "routing_density",
    "avg_scheduling_target",
]

corr = df[corr_cols].corr()

plt.figure(figsize=(8,6))

plt.imshow(corr)

plt.xticks(range(len(corr_cols)), corr_cols, rotation=45)
plt.yticks(range(len(corr_cols)), corr_cols)

plt.colorbar()

plt.title("Feature Correlation Matrix")

save_plot("correlation_matrix")


# ============================================================
# Final report
# ============================================================

print("\n" + "=" * 80)
print("EDA COMPLETE")
print("Plots saved to:")
print(OUT_DIR)

print("\nGenerated plots:")
for p in sorted(OUT_DIR.glob("*.png")):
    print("-", p.name)