import numpy as np
import pandas as pd
from pathlib import Path

train_dir = Path("data/train")
labels    = pd.read_csv("data/train_labels.csv", index_col="id")
paths     = sorted(train_dir.glob("*.csv"))

all_coords, all_true = [], []
for path in paths:
    coords = pd.read_csv(path)[["x", "y", "z"]].to_numpy(dtype=np.float64)
    all_coords.append(coords)
    all_true.append(labels.loc[path.stem].to_numpy(dtype=np.float64))

all_coords = np.stack(all_coords)  # (N, 11, 3)
all_true   = np.stack(all_true)    # (N, 3)

# 속도 시퀀스: (N, 10, 3)
vels = np.diff(all_coords, axis=1)

# 기존 물리 베이스라인
last  = all_coords[:, -1, :]
prev  = all_coords[:, -2, :]
pprev = all_coords[:, -3, :]
v_raw = last - prev
a_raw = last - 2*prev + pprev
pred_baseline = last + 2*v_raw + 0.6*a_raw
hit_baseline  = np.mean(np.linalg.norm(pred_baseline - all_true, axis=-1) <= 0.01)
print(f"Baseline (β=0.6): {hit_baseline*100:.2f}%\n")

# 방법 1: 지수 감쇠 가중 평균 속도
print("[ 지수 감쇠 가중 속도 ]")
print(f"{'decay':>6}  {'R-Hit':>8}")
best_hit, best_cfg = 0, None

for decay in np.arange(0.1, 2.01, 0.1):
    # 최근 step에 높은 가중치: w[i] = exp(decay * i), i=0(oldest)~9(newest)
    w = np.exp(decay * np.arange(10))
    w = w / w.sum()                          # (10,)
    v_weighted = (vels * w[np.newaxis, :, np.newaxis]).sum(axis=1)  # (N, 3)
    pred = last + 2 * v_weighted
    hit  = np.mean(np.linalg.norm(pred - all_true, axis=-1) <= 0.01)
    if hit > best_hit:
        best_hit, best_cfg = hit, ("exp", decay)
    print(f"{decay:6.1f}  {hit*100:7.2f}%")

# 방법 2: 지수 감쇠 가중 속도 + 가속도 보정
print("\n[ 지수 감쇠 가중 속도 + β 보정 ]")
print(f"{'decay':>6}  {'β':>5}  {'R-Hit':>8}")
for decay in [0.3, 0.5, 0.7, 1.0, 1.5]:
    w = np.exp(decay * np.arange(10))
    w = w / w.sum()
    v_weighted = (vels * w[np.newaxis, :, np.newaxis]).sum(axis=1)
    for beta in np.arange(0.0, 1.01, 0.1):
        pred = last + 2 * v_weighted + beta * a_raw
        hit  = np.mean(np.linalg.norm(pred - all_true, axis=-1) <= 0.01)
        if hit > best_hit:
            best_hit, best_cfg = hit, ("exp+beta", decay, beta)
        if abs(beta - round(beta, 1)) < 1e-9:
            print(f"{decay:6.1f}  {beta:5.1f}  {hit*100:7.2f}%")

# 방법 3: 선형 회귀로 속도 추정 (전체 시점 fitting)
print("\n[ 선형 회귀 외삽 ]")
t = np.arange(11, dtype=np.float64)  # 0~10
t_centered = t - t.mean()
# 각 샘플에 대해 x, y, z 각각 선형 회귀
# slope = Σ(t*x) / Σ(t²)
denom = (t_centered ** 2).sum()
slopes = np.einsum('ni,t->ni', np.ones((len(all_coords), 3)), np.zeros(3))
for dim in range(3):
    coords_dim = all_coords[:, :, dim]  # (N, 11)
    slope = (coords_dim * t_centered[np.newaxis, :]).sum(axis=1) / denom  # (N,)
    slopes[:, dim] = slope

# slope는 좌표/step = 좌표/40ms, 80ms = 2 steps
pred_lr = last + 2 * slopes
hit_lr  = np.mean(np.linalg.norm(pred_lr - all_true, axis=-1) <= 0.01)
print(f"Linear regression: {hit_lr*100:.2f}%")
if hit_lr > best_hit:
    best_hit, best_cfg = hit_lr, "linear_regression"

print(f"\nBest overall: {best_hit*100:.2f}%  config={best_cfg}")
