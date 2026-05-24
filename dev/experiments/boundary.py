"""
Boundary residual MLP — small correction for predictions in the 0.5-2.5 cm error zone.
Trained on OOF predictions from K-Fold selector; capped at CORRECTION_CAP.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import BOUNDARY_LO, BOUNDARY_HI, CORRECTION_CAP, EPS, BATCH_SIZE, OUTPUT_DIR
from candidates import motion_terms, make_seq_features

BOUNDARY_FEAT_DIM = 12   # seq context (9) + pred offset in Frenet (3)


class BoundaryMLP(nn.Module):
    def __init__(self, in_dim: int = BOUNDARY_FEAT_DIM, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 3),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:  # (B, 3)
        raw = self.net(feat)
        mag   = torch.norm(raw, dim=-1, keepdim=True).clamp(min=EPS)
        scale = torch.clamp(mag, max=CORRECTION_CAP) / mag
        return raw * scale


def make_boundary_features(
    coords: np.ndarray,   # (N, 11, 3)
    pred:   np.ndarray,   # (N, 3)
) -> np.ndarray:          # (N, BOUNDARY_FEAT_DIM)
    seq_feats = make_seq_features(coords)           # (N, 11, 9)
    last_feat = seq_feats[:, -1, :]                 # (N, 9)

    p0, d1, _, _, _ = motion_terms(coords, end_idx=10)
    speed   = np.linalg.norm(d1, axis=1, keepdims=True) + EPS   # (N, 1)
    tangent = d1 / speed                                          # (N, 3)

    horizon_dist = speed[:, 0] * 2.0 + EPS          # (N,)  expected step size

    delta  = pred - p0                               # (N, 3)
    par    = np.sum(delta * tangent, axis=1)         # (N,)
    perp_v = delta - par[:, np.newaxis] * tangent   # (N, 3)
    perp   = np.linalg.norm(perp_v, axis=1)         # (N,)
    dist   = np.linalg.norm(delta, axis=1)           # (N,)

    pred_feats = np.stack(
        [par / horizon_dist, perp / horizon_dist, dist / horizon_dist], axis=1
    )  # (N, 3)

    return np.concatenate([last_feat, pred_feats], axis=1).astype(np.float32)


def train_boundary(
    oof_preds:  np.ndarray,    # (N, 3) OOF selector predictions
    oof_coords: np.ndarray,    # (N, 11, 3)
    oof_labels: np.ndarray,    # (N, 3)
    device:     torch.device,
    epochs:     int   = 300,
    lr:         float = 1e-3,
) -> "BoundaryMLP | None":
    errors = np.linalg.norm(oof_preds - oof_labels, axis=-1)
    mask   = (errors >= BOUNDARY_LO) & (errors <= BOUNDARY_HI)
    print(f"Boundary samples: {mask.sum()} / {len(mask)}  ({mask.mean():.1%})")

    if mask.sum() < 32:
        print("Too few boundary samples — skipping.")
        return None

    feat_np   = make_boundary_features(oof_coords[mask], oof_preds[mask])
    target_np = (oof_labels[mask] - oof_preds[mask]).astype(np.float32)

    # Cap targets so the MLP learns only what it can actually achieve
    mags   = np.linalg.norm(target_np, axis=-1, keepdims=True)
    scales = np.minimum(mags, CORRECTION_CAP) / (mags + EPS)
    target_np = (target_np * scales).astype(np.float32)

    ds     = TensorDataset(torch.tensor(feat_np), torch.tensor(target_np))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = BoundaryMLP().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_loss, best_state = float("inf"), None
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.mse_loss(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        sched.step()
        avg = total / len(ds)
        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"Boundary MLP done — best MSE: {best_loss:.6f}")
    model.load_state_dict(best_state)
    return model


def apply_boundary(
    model:  "BoundaryMLP | None",
    coords: np.ndarray,   # (N, 11, 3)
    pred:   np.ndarray,   # (N, 3)
    device: torch.device,
) -> np.ndarray:          # (N, 3)
    if model is None:
        return pred

    feat_t = torch.tensor(make_boundary_features(coords, pred)).to(device)
    model.eval()
    with torch.no_grad():
        correction = model(feat_t).cpu().numpy()
    return pred + correction
