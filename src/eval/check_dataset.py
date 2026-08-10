import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "CCL_Simulator"))

import torch
import numpy as np


DATA_PATH = ROOT / "data/raw/simnet_samples.pt"

data = torch.load(DATA_PATH, weights_only=False)

times = np.array([s["completion_time"] for s in data], dtype=float)

print("num samples:", len(data))
print("min time:", times.min())
print("max time:", times.max())
print("mean time:", times.mean())
print("std time:", times.std())
print("num zero:", np.sum(times == 0))
print("num nan:", np.isnan(times).sum())
print("num inf:", np.isinf(times).sum())

print("\nTopology types:")
print(Counter(s["topology_type"] for s in data))

print("\nPolicy types:")
print(Counter(s["policy_type"] for s in data))

print("\nNum nodes:")
print(Counter(s["num_nodes"] for s in data))

print("\nChunk MB:")
print(Counter(s["chunk_mb"] for s in data))

print("\nFirst sample keys:")
print(data[0].keys())