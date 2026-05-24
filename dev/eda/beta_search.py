import numpy as np
import pandas as pd
from pathlib import Path

train_dir = Path("data/train")
labels    = pd.read_csv("data/train_labels.csv", index_col="id")
paths     = sorted(train_dir.glob("*.csv"))

# 전체 좌표 + 정답 로딩
all_coords, all_true = [], []
for path in paths:
    coords = pd.read_csv(path)[["x", "y", "z"]].to_numpy(dtype=np.float64)
    all_coords.append(coords)
    all_true.append(labels.loc[path.stem].to_numpy(dtype=np.float64))

all_coords = np.stack(all_coords)  # (N, 11, 3)
all_true   = np.stack(all_true)    # (N, 3)

last  = all_coords[:, -1, :]   # (N, 3)
prev  = all_coords[:, -2, :]
pprev = all_coords[:, -3, :]
v = last - prev
a = last - 2*prev + pprev

# β 그리드 서치
betas = np.arange(-1.0, 2.01, 0.05)
results = []

for beta in betas:
    pred = last + 2*v + beta * a
    hit  = np.mean(np.linalg.norm(pred - all_true, axis=-1) <= 0.01)
    results.append((beta, hit))

results.sort(key=lambda x: -x[1])

print("[ Top 15 β values ]")
print(f"{'β':>6}  {'R-Hit':>8}")
for beta, hit in results[:15]:
    print(f"{beta:6.2f}  {hit*100:7.2f}%")

print()
best_beta, best_hit = results[0]
print(f"Best β = {best_beta:.2f}  →  {best_hit*100:.2f}%")

# 더 세밀하게: best 근방 탐색
print("\n[ Fine search around best β ]")
fine_betas = np.arange(best_beta - 0.1, best_beta + 0.11, 0.01)
fine_results = []
for beta in fine_betas:
    pred = last + 2*v + beta * a
    hit  = np.mean(np.linalg.norm(pred - all_true, axis=-1) <= 0.01)
    fine_results.append((beta, hit))

fine_results.sort(key=lambda x: -x[1])
print(f"{'β':>6}  {'R-Hit':>8}")
for beta, hit in fine_results[:10]:
    print(f"{beta:6.3f}  {hit*100:.2f}%")
