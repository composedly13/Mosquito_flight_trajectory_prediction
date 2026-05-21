import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from tqdm import tqdm
import hashlib

from config import *
from dataset import load_all, MosquitoDataset, augment_batch_gpu
from model import CandidateSelector, soft_labels, selector_predict
from candidates import N_CANDIDATES, make_candidates_gpu, make_seq_features_gpu, make_cand_features_gpu
from boundary import train_boundary, apply_boundary


def r_hit(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(pred - true, axis=-1) <= R_HIT_THRESHOLD))


def fold_id(sample_id: str, n_folds: int = N_FOLDS) -> int:
    return int(hashlib.md5(sample_id.encode()).hexdigest()[:8], 16) % n_folds


def pairwise_loss(logits: torch.Tensor, soft: torch.Tensor, margin: float = 0.12) -> torch.Tensor:
    """Ranking loss: good candidates should score higher than bad ones."""
    good = (soft > 0.1).float()
    bad  = (soft < 0.01).float()
    score_good = (logits * good).sum(dim=-1) / (good.sum(dim=-1) + 1e-8)
    score_bad  = (logits * bad).sum(dim=-1)  / (bad.sum(dim=-1)  + 1e-8)
    return F.relu(margin - score_good + score_bad).mean()


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

    model     = CandidateSelector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_hit, best_state, patience_cnt = 0.0, None, 0
    best_preds = np.zeros((val_mask.sum(), 3), dtype=np.float32)

    pbar = tqdm(range(1, EPOCHS + 1), desc=f"Fold {fold}")
    for epoch in pbar:
        # Train
        model.train()
        for batch in train_loader:
            coords_b = batch["coords"].to(device, non_blocking=True)
            true     = batch["label"].to(device, non_blocking=True)

            coords_b, true = augment_batch_gpu(coords_b, true)

            cands  = make_candidates_gpu(coords_b)
            seq_f  = make_seq_features_gpu(coords_b)
            cand_f = make_cand_features_gpu(coords_b, cands)

            logits = model(seq_f, cand_f)                        # (B, C)
            soft   = soft_labels(cands, true)                    # (B, C)

            loss_ce   = -(soft * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            loss_pair = pairwise_loss(logits, soft)
            loss      = loss_ce + 0.25 * loss_pair

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # Val
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                coords_b = batch["coords"].to(device, non_blocking=True)

                cands  = make_candidates_gpu(coords_b)
                seq_f  = make_seq_features_gpu(coords_b)
                cand_f = make_cand_features_gpu(coords_b, cands)

                logits = model(seq_f, cand_f)
                pred   = selector_predict(logits, cands)
                preds.append(pred.cpu().numpy())
                trues.append(batch["label"].cpu().numpy())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        hit   = r_hit(preds, trues)

        if hit > best_hit:
            best_hit   = hit
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_preds = preds.copy()
            patience_cnt = 0
        else:
            patience_cnt += 1

        pbar.set_postfix(hit=f"{hit:.4f}", best=f"{best_hit:.4f}", patience=patience_cnt)

        if patience_cnt >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    return best_hit, best_state, val_mask, best_preds


def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Candidates: {N_CANDIDATES}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ids, coords, labels = load_all(TRAIN_DIR, LABELS_PATH)

    fold_results = []
    all_states   = []
    oof_preds    = np.zeros((len(ids), 3), dtype=np.float32)

    for fold in range(N_FOLDS):
        print(f"\n=== Fold {fold + 1}/{N_FOLDS} ===")
        hit, state, val_mask, val_preds = train_fold(fold, ids, coords, labels, device)
        fold_results.append(hit)
        all_states.append(state)
        oof_preds[val_mask] = val_preds
        print(f"Fold {fold + 1} best R-Hit: {hit:.4f}")

    print(f"\n{'='*40}")
    print(f"CV mean R-Hit: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    for i, h in enumerate(fold_results):
        print(f"  Fold {i+1}: {h:.4f}")

    oof_hit = r_hit(oof_preds, labels)
    print(f"OOF R-Hit (selector only): {oof_hit:.4f}")

    # Save selector models
    for i, state in enumerate(all_states):
        torch.save(state, OUTPUT_DIR / f"selector_fold{i}.pt")

    # Train and save boundary MLP
    print("\n=== Boundary MLP ===")
    boundary_model = train_boundary(oof_preds, coords, labels, device)
    if boundary_model is not None:
        torch.save(boundary_model.state_dict(), OUTPUT_DIR / "boundary.pt")
        corrected = apply_boundary(boundary_model, coords, oof_preds, device)
        corrected_hit = r_hit(corrected, labels)
        print(f"OOF R-Hit (with boundary): {corrected_hit:.4f}")

    print(f"\nModels saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    train()
