import numpy as np
import pandas as pd
from pathlib import Path

train_dir = Path("data/train")
labels = pd.read_csv("data/train_labels.csv", index_col="id")

results = []

for path in sorted(train_dir.glob("*.csv")):
    df = pd.read_csv(path)
    sid = path.stem
    coords = df[["x", "y", "z"]].to_numpy()

    last, prev, pprev = coords[-1], coords[-2], coords[-3]
    v = last - prev
    a = last - 2 * prev + pprev

    speed     = np.linalg.norm(v)
    accel_mag = np.linalg.norm(a)

    pred_linear = last + 2 * v
    pred_accel  = last + 2 * v + 0.6 * a

    true = labels.loc[sid].to_numpy()

    err_lin   = np.linalg.norm(pred_linear - true)
    err_accel = np.linalg.norm(pred_accel  - true)

    # 전체 궤적 직선성 (첫-끝 거리 / 실제 이동 거리)
    total_dist = sum(np.linalg.norm(coords[i+1] - coords[i]) for i in range(len(coords)-1))
    straight   = np.linalg.norm(coords[-1] - coords[0])
    linearity  = straight / total_dist if total_dist > 0 else 1.0

    results.append({
        "id": sid,
        "speed": speed,
        "accel_mag": accel_mag,
        "err_linear": err_lin,
        "err_accel": err_accel,
        "hit_linear": err_lin <= 0.01,
        "hit_accel": err_accel <= 0.01,
        "linearity": linearity,
        "total_dist": total_dist,
    })

df_res = pd.DataFrame(results)

print("=" * 50)
print(f"총 샘플 수: {len(df_res)}")
print()

print("[ 적중률 ]")
print(f"  선형 외삽     : {df_res['hit_linear'].mean()*100:.2f}%")
print(f"  가속도(β=0.6) : {df_res['hit_accel'].mean()*100:.2f}%")
print()

print("[ 오차 분포 (cm) ]")
for col, label in [("err_linear", "선형"), ("err_accel", "가속도")]:
    e = df_res[col] * 100
    print(f"  {label}  mean={e.mean():.3f}  median={e.median():.3f}  "
          f"p75={e.quantile(.75):.3f}  p95={e.quantile(.95):.3f}  max={e.max():.3f}")
print()

print("[ 속도 분포 (m/40ms) ]")
s = df_res["speed"]
print(f"  mean={s.mean():.4f}  median={s.median():.4f}  "
      f"p95={s.quantile(.95):.4f}  max={s.max():.4f}")
print()

print("[ 가속도 크기 분포 (m/40ms²) ]")
a = df_res["accel_mag"]
print(f"  mean={a.mean():.5f}  median={a.median():.5f}  "
      f"p95={a.quantile(.95):.5f}  max={a.max():.5f}")
print()

print("[ 궤적 직선성 (1.0 = 완전 직선) ]")
l = df_res["linearity"]
print(f"  mean={l.mean():.4f}  median={l.median():.4f}  "
      f"p5={l.quantile(.05):.4f}  min={l.min():.4f}")
print()

# 선형은 HIT이지만 가속도는 MISS인 케이스
lin_only = df_res[df_res["hit_linear"] & ~df_res["hit_accel"]]
accel_only = df_res[~df_res["hit_linear"] & df_res["hit_accel"]]
both_miss = df_res[~df_res["hit_linear"] & ~df_res["hit_accel"]]
print("[ 케이스 분류 ]")
print(f"  둘 다 HIT           : {(df_res['hit_linear'] & df_res['hit_accel']).sum()}")
print(f"  선형만 HIT          : {len(lin_only)}")
print(f"  가속도만 HIT        : {len(accel_only)}")
print(f"  둘 다 MISS          : {len(both_miss)}")
print()

print("[ 둘 다 MISS인 샘플의 속도/가속도 특성 ]")
print(f"  speed    mean={both_miss['speed'].mean():.4f}  (전체 {df_res['speed'].mean():.4f})")
print(f"  accel    mean={both_miss['accel_mag'].mean():.5f}  (전체 {df_res['accel_mag'].mean():.5f})")
print(f"  linearity mean={both_miss['linearity'].mean():.4f}  (전체 {df_res['linearity'].mean():.4f})")
