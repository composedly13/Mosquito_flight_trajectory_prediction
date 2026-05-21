import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from candidates import make_candidates, make_seq_features, make_cand_features, N_CANDIDATES


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


class MosquitoDataset(Dataset):
    def __init__(self, coords: np.ndarray, labels: np.ndarray = None, augment: bool = False):
        self.coords  = coords
        self.labels  = labels
        self.augment = augment
        self.is_train = labels is not None

    def __len__(self):
        return len(self.coords)

    def _augment(self, coords: np.ndarray, label: np.ndarray = None):
        """Random 3D rotation."""
        u  = np.random.uniform(0, 1, 3).astype(np.float32)
        q  = np.array([
            np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1]),
            np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1]),
            np.sqrt(u[0])     * np.sin(2 * np.pi * u[2]),
            np.sqrt(u[0])     * np.cos(2 * np.pi * u[2]),
        ], dtype=np.float32)
        w, x, y, z = q
        R = np.array([
            [1-2*(y*y+z*z),  2*(x*y-w*z),   2*(x*z+w*y)],
            [2*(x*y+w*z),   1-2*(x*x+z*z),  2*(y*z-w*x)],
            [2*(x*z-w*y),    2*(y*z+w*x),  1-2*(x*x+y*y)],
        ], dtype=np.float32)
        coords = coords @ R.T
        if label is not None:
            label = label @ R.T
        return coords, label

    def __getitem__(self, idx):
        coords = self.coords[idx].copy()
        label  = self.labels[idx].copy() if self.is_train else None

        if self.augment:
            coords, label = self._augment(coords, label)

        cands     = make_candidates(coords[np.newaxis])[0]       # (C, 3)
        seq_feat  = make_seq_features(coords[np.newaxis])[0]     # (11, 9)
        cand_feat = make_cand_features(
            coords[np.newaxis], cands[np.newaxis]
        )[0]                                                      # (C, 10)

        out = {
            "seq_feat":  torch.tensor(seq_feat),
            "cand_feat": torch.tensor(cand_feat),
            "cands":     torch.tensor(cands),
        }
        if self.is_train:
            out["label"] = torch.tensor(label)
        return out
