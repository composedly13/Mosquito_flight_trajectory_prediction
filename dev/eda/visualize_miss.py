import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

train_dir = Path("data/train")
labels = pd.read_csv("data/train_labels.csv", index_col="id")

# 전체 샘플 분류
hits, misses = [], []

for path in sorted(train_dir.glob("*.csv")):
    df = pd.read_csv(path)
    sid = path.stem
    coords = df[["x", "y", "z"]].to_numpy()
    last, prev = coords[-1], coords[-2]
    v = last - prev
    pred = last + 2 * v
    true = labels.loc[sid].to_numpy()
    err = np.linalg.norm(pred - true)
    entry = {"id": sid, "path": path, "err": err}
    (hits if err <= 0.01 else misses).append(entry)

misses_sorted = sorted(misses, key=lambda x: x["err"])
samples = {
    "HIT (easy)":   hits[:3],
    "MISS (medium)":  misses_sorted[len(misses)//2 : len(misses)//2 + 3],
    "MISS (hard)": misses_sorted[-3:],
}

fig = plt.figure(figsize=(18, 12))
fig.suptitle("Trajectory Visualization: HIT vs MISS", fontsize=14)

col = 0
for group_name, group in samples.items():
    for item in group:
        df = pd.read_csv(item["path"])
        coords = df[["x", "y", "z"]].to_numpy()
        sid = item["id"]

        last, prev = coords[-1], coords[-2]
        v = last - prev
        pred_linear = last + 2 * v
        true = labels.loc[sid].to_numpy()

        ax = fig.add_subplot(3, 3, col + 1, projection="3d")
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2],
                "b-o", markersize=3, label="observed")
        ax.scatter(*pred_linear, c="orange", s=80, marker="^", label="predicted")
        ax.scatter(*true,        c="red",    s=80, marker="*", label="actual")
        ax.set_title(f"{group_name}\n{sid} | err={item['err']*100:.2f}cm", fontsize=8)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        if col == 0:
            ax.legend(fontsize=7)
        col += 1

plt.tight_layout()
plt.savefig("dev/eda/miss_visualization.png", dpi=150)
print("Saved: dev/eda/miss_visualization.png")
plt.show()
