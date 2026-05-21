import numpy as np
import pandas as pd
from pathlib import Path

samples = [f"data/train/TRAIN_0000{i}.csv" for i in range(1, 6)]
labels = pd.read_csv("data/train_labels.csv", index_col="id")

for path in samples:
    df = pd.read_csv(path)
    sid = Path(path).stem
    coords = df[["x", "y", "z"]].to_numpy()
    last, prev, pprev = coords[-1], coords[-2], coords[-3]

    v = last - prev
    a = last - 2 * prev + pprev

    pred_linear = last + 2 * v
    pred_accel  = last + 2 * v + 0.6 * a

    true = labels.loc[sid].to_numpy()

    err_lin   = np.linalg.norm(pred_linear - true)
    err_accel = np.linalg.norm(pred_accel  - true)

    hit_lin   = "HIT " if err_lin   <= 0.01 else "MISS"
    hit_accel = "HIT " if err_accel <= 0.01 else "MISS"

    print(f"{sid}")
    print(f"  v          : {v.round(5)}")
    print(f"  a          : {a.round(5)}")
    print(f"  linear     : {err_lin*100:.3f} cm  [{hit_lin}]")
    print(f"  accel(0.6) : {err_accel*100:.3f} cm  [{hit_accel}]")
    print()
