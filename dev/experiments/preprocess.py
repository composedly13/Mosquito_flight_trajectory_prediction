import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from config import BETA


def physics_predict(coords: np.ndarray, beta: float = BETA) -> np.ndarray:
    """x(0) + 2v + beta*a"""
    last  = coords[-1]
    prev  = coords[-2]
    pprev = coords[-3]
    v = last - prev
    a = last - 2 * prev + pprev
    return last + 2 * v + beta * a


def extract_features(coords: np.ndarray) -> np.ndarray:
    """
    coords: (11, 3)
    returns: (11, 9) = rel_coords(3) + velocity(3) + acceleration(3)
    """
    last = coords[-1]
    rel  = coords - last                                          # (11, 3) 상대 좌표

    vels  = np.diff(rel, axis=0)                                 # (10, 3)
    accels = np.diff(vels, axis=0)                               # (9, 3)

    vel_pad   = np.vstack([np.zeros((1, 3), dtype=np.float32), vels])
    accel_pad = np.vstack([np.zeros((2, 3), dtype=np.float32), accels])

    return np.concatenate([rel, vel_pad, accel_pad], axis=1).astype(np.float32)  # (11, 9)


class MosquitoDataset(Dataset):
    def __init__(self, data_dir: Path, labels_path: Path = None, beta: float = BETA):
        paths = sorted(data_dir.glob("*.csv"))
        self.is_train = labels_path is not None

        if self.is_train:
            labels = pd.read_csv(labels_path, index_col="id")

        print(f"Loading {len(paths)} samples...")
        xs, physics_preds, last_coords, targets = [], [], [], []

        for path in paths:
            coords = pd.read_csv(path)[["x", "y", "z"]].to_numpy(dtype=np.float32)
            last   = coords[-1]
            phys   = physics_predict(coords, beta)

            xs.append(extract_features(coords))
            physics_preds.append(phys)
            last_coords.append(last)

            if self.is_train:
                true = labels.loc[path.stem].to_numpy(dtype=np.float32)
                # 타겟: 물리 예측 대비 보정값 (correction)
                targets.append(true - phys)

        self.X       = torch.tensor(np.stack(xs))             # (N, 11, 9)
        self.physics = torch.tensor(np.stack(physics_preds))  # (N, 3)
        self.last    = torch.tensor(np.stack(last_coords))    # (N, 3)

        if self.is_train:
            self.Y = torch.tensor(np.stack(targets))          # (N, 3) correction

        print("Done.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.is_train:
            return self.X[idx], self.physics[idx], self.Y[idx]
        return self.X[idx], self.physics[idx]
