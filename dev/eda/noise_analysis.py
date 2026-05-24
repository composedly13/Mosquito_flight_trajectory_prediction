import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter

train_dir = Path("data/train")
labels    = pd.read_csv("data/train_labels.csv", index_col="id")
paths     = sorted(train_dir.glob("*.csv"))[:10000]

raw_errors, sg_errors = [], []
noise_levels = []

for path in paths:
    coords = pd.read_csv(path)[["x", "y", "z"]].to_numpy(dtype=np.float64)
    sid    = path.stem
    true   = labels.loc[sid].to_numpy(dtype=np.float64)

    # 노이즈 추정: 2차 차분의 표준편차 (가속도 변화량)
    noise = np.std(np.diff(coords, n=2, axis=0))
    noise_levels.append(noise)

    # 물리 예측 (raw)
    v    = coords[-1] - coords[-2]
    a    = coords[-1] - 2*coords[-2] + coords[-3]
    pred_raw = coords[-1] + 2*v + 0.6*a

    # Savitzky-Golay 스무딩 (window=7, poly=3)
    if len(coords) >= 7:
        smoothed = savgol_filter(coords, window_length=7, polyorder=3, axis=0)
    else:
        smoothed = coords

    v_sg = smoothed[-1] - smoothed[-2]
    a_sg = smoothed[-1] - 2*smoothed[-2] + smoothed[-3]
    pred_sg = smoothed[-1] + 2*v_sg + 0.6*a_sg

    raw_errors.append(np.linalg.norm(pred_raw - true))
    sg_errors.append(np.linalg.norm(pred_sg   - true))

raw_errors = np.array(raw_errors)
sg_errors  = np.array(sg_errors)
noise_levels = np.array(noise_levels)

print("[ 노이즈 수준 (2차 차분 std) ]")
print(f"  mean={noise_levels.mean():.6f}  median={noise_levels.median() if hasattr(noise_levels,'median') else np.median(noise_levels):.6f}  max={noise_levels.max():.6f}")
print()
print("[ R-Hit@1cm ]")
print(f"  Raw physics  : {np.mean(raw_errors <= 0.01)*100:.2f}%")
print(f"  SG smoothed  : {np.mean(sg_errors  <= 0.01)*100:.2f}%")
print()
print("[ 오차 분포 (cm) ]")
for name, errs in [("Raw", raw_errors), ("SG", sg_errors)]:
    print(f"  {name}  mean={errs.mean()*100:.3f}  median={np.median(errs)*100:.3f}  p95={np.percentile(errs,95)*100:.3f}")
print()

# SG가 더 나은 케이스 vs 더 나쁜 케이스
sg_better = np.sum(sg_errors < raw_errors)
sg_worse  = np.sum(sg_errors > raw_errors)
print(f"[ SG 효과 ]")
print(f"  SG가 더 좋음: {sg_better} ({sg_better/len(paths)*100:.1f}%)")
print(f"  SG가 더 나쁨: {sg_worse}  ({sg_worse/len(paths)*100:.1f}%)")
print(f"  동일        : {len(paths) - sg_better - sg_worse}")
