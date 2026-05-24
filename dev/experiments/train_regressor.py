"""
TransformerRegressor 학습 스크립트.

Strategy:
  - CandidateSelector와 동일한 fold 분할 (MD5 hash)
  - Input: seq_features only (11×11), Output: 3D position
  - Loss: L2 distance mean (= R-Hit@1cm 간접 최적화)
  - Early stopping on validation distance (lower is better)
  - Save: outputs/seed{seed}/reg2_fold{i}.pt

Evaluation:
  - OOF R-Hit@1cm (overall)
  - C-group R-Hit@1cm (samples with no candidate within 1cm)
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import hashlib

from config import (
    TRAIN_DIR, LABELS_PATH, OUTPUT_DIR,
    SEED, N_FOLDS, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, PATIENCE,
    AUG_MODE, SPEED_SCALE_RANGE, SPEED_SCALE_PROB,
    R_HIT_THRESHOLD,
)
from dataset import (
    load_all, MosquitoDataset,
    augment_batch_gpu, augment_batch_gpu_yaw, augment_speed_scale_gpu,
)
from model import TransformerRegressor
from candidates import (
    N_CANDIDATES, make_candidates,
    make_seq_features_gpu,
)


def r_hit(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(pred - true, axis=-1) <= R_HIT_THRESHOLD))


def fold_id(sample_id: str, n_folds: int = N_FOLDS) -> int:
    return int(hashlib.md5(sample_id.encode()).hexdigest()[:8], 16) % n_folds


def train_fold(
    fold: int,
    ids: list,
    coords: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
):
    val_mask   = np.array([fold_id(i) == fold for i in ids])
    train_mask = ~val_mask

    train_ds = MosquitoDataset(coords[train_mask], labels[train_mask], augment=True)
    val_ds   = MosquitoDataset(coords[val_mask],   labels[val_mask],   augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model     = TransformerRegressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_dist, best_state, patience_cnt = float("inf"), None, 0
    best_preds = np.zeros((val_mask.sum(), 3), dtype=np.float32)

    pbar = tqdm(range(1, EPOCHS + 1), desc=f"Fold {fold}")
    for epoch in pbar:
        # ---------- Train ----------
        model.train()
        train_dist_sum, train_n = 0.0, 0
        for batch in train_loader:
            coords_b = batch["coords"].to(device, non_blocking=True)
            true     = batch["label"].to(device, non_blocking=True)

            # Augmentation (same as selector)
            if AUG_MODE == 'so3':
                coords_b, true = augment_batch_gpu(coords_b, true)
            elif AUG_MODE == 'yaw':
                coords_b, true = augment_batch_gpu_yaw(coords_b, true)
            elif AUG_MODE == 'yaw_speed':
                coords_b, true = augment_batch_gpu_yaw(coords_b, true)
                coords_b, true = augment_speed_scale_gpu(
                    coords_b, true,
                    scale_range=SPEED_SCALE_RANGE,
                    prob=SPEED_SCALE_PROB,
                )

            seq_f = make_seq_features_gpu(coords_b)  # (B, 11, 11)
            p0    = coords_b[:, -1, :]               # (B, 3) 마지막 알려진 위치
            pred  = model(seq_f, p0)                 # (B, 3) = p0 + offset

            # L2 distance loss — directly minimizes mean distance
            dist = torch.norm(pred - true, dim=-1)   # (B,)
            loss = dist.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_dist_sum += dist.sum().item()
            train_n        += len(true)

        scheduler.step()

        # ---------- Val ----------
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                coords_b = batch["coords"].to(device, non_blocking=True)
                seq_f    = make_seq_features_gpu(coords_b)
                p0       = coords_b[:, -1, :]
                pred     = model(seq_f, p0)
                preds.append(pred.cpu().numpy())
                trues.append(batch["label"].cpu().numpy())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        val_dist = float(np.linalg.norm(preds - trues, axis=-1).mean())
        val_hit  = r_hit(preds, trues)

        if val_dist < best_dist:
            best_dist  = val_dist
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_preds = preds.copy()
            patience_cnt = 0
        else:
            patience_cnt += 1

        pbar.set_postfix(
            dist=f"{val_dist:.5f}",
            best=f"{best_dist:.5f}",
            hit=f"{val_hit:.4f}",
            p=patience_cnt,
        )

        if patience_cnt >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    return best_dist, best_state, val_mask, best_preds


def train(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Seed: {seed}  |  Aug: {AUG_MODE}")
    print(f"Loss: L2 distance  |  Model: TransformerRegressor")

    out_dir = OUTPUT_DIR / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ids, coords, labels = load_all(TRAIN_DIR, LABELS_PATH)

    # C-group mask: no candidate within 1cm (physics-based, independent of selector)
    print("Computing C-group mask (no oracle candidate within 1cm)...")
    oracle_cands = make_candidates(coords)   # (N, C, 3)
    min_dists    = np.linalg.norm(
        oracle_cands - labels[:, np.newaxis, :], axis=-1
    ).min(axis=1)                            # (N,)
    c_group_mask = min_dists > R_HIT_THRESHOLD
    print(f"C-group: {c_group_mask.sum()} / {len(ids)} ({c_group_mask.mean():.1%})")

    fold_results = []
    all_states   = []
    oof_preds    = np.zeros((len(ids), 3), dtype=np.float32)

    for fold in range(N_FOLDS):
        print(f"\n=== Fold {fold + 1}/{N_FOLDS} ===")
        dist, state, val_mask, val_preds = train_fold(fold, ids, coords, labels, device)
        fold_results.append(dist)
        all_states.append(state)
        oof_preds[val_mask] = val_preds
        print(f"Fold {fold + 1} best val-dist: {dist:.5f}")

    print(f"\n{'='*40}")
    print(f"CV mean val-dist: {np.mean(fold_results):.5f} ± {np.std(fold_results):.5f}")
    for i, d in enumerate(fold_results):
        print(f"  Fold {i+1}: {d:.5f}")

    oof_hit = r_hit(oof_preds, labels)
    print(f"\nOOF R-Hit (regressor, overall): {oof_hit:.4f}")

    # C-group specific evaluation
    c_preds = oof_preds[c_group_mask]
    c_trues = labels[c_group_mask]
    c_hit   = r_hit(c_preds, c_trues)
    c_dist  = float(np.linalg.norm(c_preds - c_trues, axis=-1).mean())
    print(f"OOF R-Hit (regressor, C-group): {c_hit:.4f}  (mean dist: {c_dist:.5f})")

    # A/B-group baseline for comparison
    ab_mask = ~c_group_mask
    ab_hit  = r_hit(oof_preds[ab_mask], labels[ab_mask])
    print(f"OOF R-Hit (regressor, A+B-group): {ab_hit:.4f}  ({ab_mask.sum()} samples)")

    # Save models
    for i, state in enumerate(all_states):
        torch.save(state, out_dir / f"reg2_fold{i}.pt")

    # Save OOF preds for entropy-blend analysis
    np.save(out_dir / "reg2_oof_preds.npy", oof_preds)
    print(f"\nModels saved to {out_dir}/")
    print(f"OOF preds saved: {out_dir}/reg2_oof_preds.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    train(seed=args.seed)
