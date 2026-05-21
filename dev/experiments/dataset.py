import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path


def load_all(data_dir: Path, labels_path: Path = None):
    paths = sorted(data_dir.glob("*.csv"))
    ids   = [p.stem for p in paths]

    print(f"Loading {len(paths)} samples...")
    coords = np.stack([
        pd.read_csv(p)[["x", "y", "z"]].to_numpy(dtype=np.float32)
        for p in paths
    ])  # (N, 11, 3)

    labels = None
    if labels_path is not None:
        df_labels = pd.read_csv(labels_path, index_col="id")
        labels = np.stack([df_labels.loc[i].to_numpy(dtype=np.float32) for i in ids])

    print("Done.")
    return ids, coords, labels


def augment_batch_gpu(coords: torch.Tensor, labels: torch.Tensor):
    """Random SO3 rotation on GPU. coords: (B,11,3), labels: (B,3)"""
    B, dev, dt = coords.shape[0], coords.device, coords.dtype
    u  = torch.rand(B, 3, device=dev, dtype=dt)
    s1 = (1 - u[:, 0]).sqrt()
    s0 = u[:, 0].sqrt()
    p2 = 2 * torch.pi

    qw = s1 * (p2 * u[:, 1]).sin()
    qx = s1 * (p2 * u[:, 1]).cos()
    qy = s0 * (p2 * u[:, 2]).sin()
    qz = s0 * (p2 * u[:, 2]).cos()

    R = torch.stack([
        1-2*(qy*qy+qz*qz),  2*(qx*qy-qw*qz),  2*(qx*qz+qw*qy),
          2*(qx*qy+qw*qz),  1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx),
          2*(qx*qz-qw*qy),    2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy),
    ], dim=-1).view(B, 3, 3)

    coords_r = torch.bmm(coords, R.mT)
    labels_r = (labels[:, None] @ R.mT).squeeze(1)
    return coords_r, labels_r


class MosquitoDataset(Dataset):
    def __init__(self, coords: np.ndarray, labels: np.ndarray = None, augment: bool = False):
        self.coords  = coords
        self.labels  = labels
        self.augment = augment  # read by train loop to decide GPU augmentation

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        out = {"coords": torch.tensor(self.coords[idx])}
        if self.labels is not None:
            out["label"] = torch.tensor(self.labels[idx])
        return out
