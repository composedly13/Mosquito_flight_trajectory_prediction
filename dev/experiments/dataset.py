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


def _build_rotation_matrices(coords: torch.Tensor):
    """Uniformly random SO3 rotation matrices. Returns R: (B, 3, 3)."""
    B, dev, dt = coords.shape[0], coords.device, coords.dtype
    u  = torch.rand(B, 3, device=dev, dtype=dt)
    s1 = (1 - u[:, 0]).sqrt()
    s0 = u[:, 0].sqrt()
    p2 = 2 * torch.pi
    qw = s1 * (p2 * u[:, 1]).sin()
    qx = s1 * (p2 * u[:, 1]).cos()
    qy = s0 * (p2 * u[:, 2]).sin()
    qz = s0 * (p2 * u[:, 2]).cos()
    return torch.stack([
        1-2*(qy*qy+qz*qz),  2*(qx*qy-qw*qz),  2*(qx*qz+qw*qy),
          2*(qx*qy+qw*qz),  1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx),
          2*(qx*qz-qw*qy),    2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy),
    ], dim=-1).view(B, 3, 3)


def augment_batch_gpu(coords: torch.Tensor, labels: torch.Tensor):
    """Random SO3 rotation on GPU. coords: (B,11,3), labels: (B,3)"""
    R = _build_rotation_matrices(coords)
    return torch.bmm(coords, R.mT), (labels[:, None] @ R.mT).squeeze(1)


def augment_batch_gpu_with_R(coords: torch.Tensor):
    """Random SO3 rotation. Returns (coords_rotated, R) — R needed to undo rotation for TTA."""
    R = _build_rotation_matrices(coords)
    return torch.bmm(coords, R.mT), R


def _build_yaw_matrices(coords: torch.Tensor):
    """Random yaw (Z-axis) rotation matrices. Preserves z=UP direction. Returns R: (B, 3, 3)."""
    B, dev, dt = coords.shape[0], coords.device, coords.dtype
    theta = torch.rand(B, device=dev, dtype=dt) * 2 * torch.pi
    cos_t = theta.cos()
    sin_t = theta.sin()
    ones  = torch.ones(B, device=dev, dtype=dt)
    zeros = torch.zeros(B, device=dev, dtype=dt)
    return torch.stack([
        cos_t, -sin_t, zeros,
        sin_t,  cos_t, zeros,
        zeros,  zeros,  ones,
    ], dim=-1).view(B, 3, 3)


def augment_batch_gpu_yaw(coords: torch.Tensor, labels: torch.Tensor):
    """Random yaw rotation on GPU. Safe for z=UP coordinate frames."""
    R = _build_yaw_matrices(coords)
    return torch.bmm(coords, R.mT), (labels[:, None] @ R.mT).squeeze(1)


def augment_batch_gpu_yaw_with_R(coords: torch.Tensor):
    """Random yaw rotation. Returns (coords_rotated, R) for TTA un-rotation."""
    R = _build_yaw_matrices(coords)
    return torch.bmm(coords, R.mT), R


def augment_mirror_gpu(coords: torch.Tensor, labels: torch.Tensor) -> tuple:
    """Independent x/y axis mirror flip (each prob=0.5). Preserves z=UP.
    - y(left) flip: left/right symmetry — always physically valid
    - x(forward) flip: forward/backward — valid assuming no sensor directional bias
    - z(up) flip: gravity axis — NEVER applied (would create upside-down trajectories)
    4 combinations: no flip / x only / y only / x+y  (each 25%)
    """
    B, dev, dt = coords.shape[0], coords.device, coords.dtype
    sign_x = (torch.rand(B, device=dev, dtype=dt) < 0.5).to(dt) * 2 - 1  # (B,) ∈ {-1,+1}
    sign_y = (torch.rand(B, device=dev, dtype=dt) < 0.5).to(dt) * 2 - 1

    coords_aug = coords.clone()
    coords_aug[:, :, 0] = coords[:, :, 0] * sign_x[:, None]
    coords_aug[:, :, 1] = coords[:, :, 1] * sign_y[:, None]

    labels_aug = labels.clone()
    labels_aug[:, 0] = labels[:, 0] * sign_x
    labels_aug[:, 1] = labels[:, 1] * sign_y
    return coords_aug, labels_aug


def augment_noise_gpu(coords: torch.Tensor, std: float = 0.001) -> torch.Tensor:
    """Gaussian coordinate noise on input trajectory only (label stays clean).
    Simulates LiDAR sensor measurement noise (~1mm).
    Encourages the model to be robust to small positional perturbations."""
    return coords + torch.randn_like(coords) * std


def augment_speed_scale_gpu(
    coords: torch.Tensor,
    labels: torch.Tensor,
    scale_range: tuple = (0.85, 1.15),
    prob: float = 0.5,
) -> tuple:
    """Speed-scale augmentation: scale all displacements relative to the last point p0.

    Simulates the mosquito moving faster or slower without changing direction.
    p0 (coords[:, 10]) remains fixed; all other points and the label scale around it.
    This encourages the selector to learn scale-invariant ranking, not absolute speed.
    """
    B, dev, dt = coords.shape[0], coords.device, coords.dtype
    lo, hi = scale_range
    scale = torch.rand(B, device=dev, dtype=dt) * (hi - lo) + lo   # (B,) ~ Uniform
    apply = (torch.rand(B, device=dev, dtype=dt) < prob).to(dt)    # (B,) Bernoulli
    scale = apply * scale + (1.0 - apply) * 1.0                    # no-op when not applied

    p0 = coords[:, 10]  # (B, 3) — last observed point, the reference
    coords_aug = p0[:, None] + scale[:, None, None] * (coords - p0[:, None])
    labels_aug = p0 + scale[:, None] * (labels - p0)
    return coords_aug, labels_aug


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
