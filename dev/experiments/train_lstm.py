"""
BiLSTM C-group regression model training.

핵심 개선 vs TransformerRegressor (reg2):
  1. Frenet-parametric 출력 (d1_scale, par, perp, jerk_scale) → yaw-invariant
     - seq_features는 회전 불변, 출력 계수도 Frenet frame → yaw 증강과 호환
     - TransformerRegressor는 raw (Δx,Δy,Δz) 출력 → yaw 증강과 비호환 (실패 원인)
  2. C-group 10× 가중 손실
     - 배치마다 make_candidates_gpu → min_dist > 1cm 샘플 = C-group
     - 기존 reg2: 전체 10K 균일 학습 → 쉬운 샘플에 수렴, C-group 무시
  3. BiLSTM (hidden=64, 2 layers)
     - 더 작은 모델 → 10K 샘플에서 과적합 감소
     - 양방향 → 궤적의 앞/뒤 맥락 동시 활용

Early stopping: 전체 val R-Hit (전체 메트릭과 정렬)
Diagnostics: C-group val R-Hit 별도 추적

Usage:
    $env:KMP_DUPLICATE_LIB_OK="TRUE"
    python dev/experiments/train_lstm.py --seed 42
    → dev/experiments/outputs/seed42/lstm_fold{0..4}.pt
    → dev/experiments/outputs/seed42/lstm_oof_preds.npy
"""
import argparse
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    TRAIN_DIR, LABELS_PATH, OUTPUT_DIR,
    SEED, N_FOLDS, BATCH_SIZE, EPOCHS, WEIGHT_DECAY,
    R_HIT_THRESHOLD,
)
from dataset import load_all, MosquitoDataset, augment_batch_gpu_yaw
from model import MosquitoLSTM
from candidates import (
    make_candidates,              # CPU: (N, C, 3) — C-group mask 전처리용
    make_candidates_gpu,          # GPU: (B, C, 3) — 배치별 C-group 감지
    make_seq_features_gpu,        # GPU: (B, 11, 11)
    N_CANDIDATES,
)

CGROUP_WEIGHT = 10.0    # C-group 손실 가중치 (selector miss 케이스)
LSTM_LR       = 1e-3    # 작은 모델 → 빠른 lr
LSTM_PATIENCE = 30      # 수렴 조건 완화 (작은 모델, 빠른 수렴)


def r_hit(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(pred - true, axis=-1) <= R_HIT_THRESHOLD))


def fold_id(sample_id: str, n_folds: int = N_FOLDS) -> int:
    return int(hashlib.md5(sample_id.encode()).hexdigest()[:8], 16) % n_folds


def train_fold(fold, ids, coords, labels, device):
    val_mask   = np.array([fold_id(i) == fold for i in ids])
    train_mask = ~val_mask

    # Val C-group 마스크 CPU에서 전처리 (학습 루프 외부, 1회)
    print(f"  C-group mask 계산 중 (fold {fold})...")
    val_oracle = make_candidates(coords[val_mask])              # (Nval, C, 3)
    val_cg_mask = (
        np.linalg.norm(val_oracle - labels[val_mask][:, None], axis=-1)
        .min(axis=1) > R_HIT_THRESHOLD
    )  # (Nval,) bool
    print(f"  Val C-group: {val_cg_mask.sum()} / {val_mask.sum()} ({val_cg_mask.mean():.1%})")

    train_ds = MosquitoDataset(coords[train_mask], labels[train_mask], augment=True)
    val_ds   = MosquitoDataset(coords[val_mask],   labels[val_mask],   augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    model     = MosquitoLSTM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LSTM_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_hit, best_state, patience_cnt = 0.0, None, 0
    best_preds = np.zeros((val_mask.sum(), 3), dtype=np.float32)

    pbar = tqdm(range(1, EPOCHS + 1), desc=f"Fold {fold}")
    for epoch in pbar:
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        for batch in train_loader:
            coords_b = batch["coords"].to(device, non_blocking=True)
            true_b   = batch["label"].to(device, non_blocking=True)

            coords_b, true_b = augment_batch_gpu_yaw(coords_b, true_b)

            seq_f = make_seq_features_gpu(coords_b)
            pred  = model(seq_f, coords_b)

            # C-group 감지: 1cm 이내 후보 없으면 C-group → 10× 가중
            with torch.no_grad():
                cands_b  = make_candidates_gpu(coords_b)          # (B, C, 3)
                min_dist = (cands_b - true_b.unsqueeze(1)).norm(dim=-1).min(dim=-1).values
                is_cg    = (min_dist > R_HIT_THRESHOLD).float()   # (B,)
                weights  = 1.0 + (CGROUP_WEIGHT - 1.0) * is_cg   # 1.0 or 10.0

            # smooth_l1: beta=R_HIT_THRESHOLD → 1cm 미만 오차는 L2, 이상은 L1
            loss_per = F.smooth_l1_loss(
                pred, true_b, beta=R_HIT_THRESHOLD, reduction='none'
            ).mean(-1)                                             # (B,)
            loss = (weights * loss_per).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # ── Val ───────────────────────────────────────────────────────────────
        model.eval()
        preds_list, trues_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                coords_b = batch["coords"].to(device, non_blocking=True)
                seq_f    = make_seq_features_gpu(coords_b)
                pred     = model(seq_f, coords_b)
                preds_list.append(pred.cpu().numpy())
                trues_list.append(batch["label"].cpu().numpy())

        preds_np = np.concatenate(preds_list)
        trues_np = np.concatenate(trues_list)

        hit    = r_hit(preds_np, trues_np)
        c_hit  = r_hit(preds_np[val_cg_mask], trues_np[val_cg_mask])
        ab_hit = r_hit(preds_np[~val_cg_mask], trues_np[~val_cg_mask])

        if hit > best_hit:
            best_hit   = hit
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_preds = preds_np.copy()
            patience_cnt = 0
        else:
            patience_cnt += 1

        pbar.set_postfix(
            hit=f"{hit:.4f}",
            c_hit=f"{c_hit:.4f}",
            best=f"{best_hit:.4f}",
            p=patience_cnt,
        )

        if patience_cnt >= LSTM_PATIENCE:
            print(f"  Early stop @ epoch {epoch}")
            break

    return best_hit, best_state, val_mask, best_preds, val_cg_mask


def train(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Seed: {seed}  |  C-group weight: {CGROUP_WEIGHT}×")
    print(f"Model: BiLSTM(hidden=64, layers=2, bidirectional)  |  Frenet-parametric output")
    print(f"Loss: C-group {CGROUP_WEIGHT}× weighted smooth_l1 (beta={R_HIT_THRESHOLD}m)")

    out_dir = OUTPUT_DIR / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ids, coords, labels = load_all(TRAIN_DIR, LABELS_PATH)

    # 전체 C-group 마스크 (진단용)
    print("전체 C-group 마스크 계산 중...")
    oracle_cands  = make_candidates(coords)
    min_dists     = np.linalg.norm(
        oracle_cands - labels[:, np.newaxis, :], axis=-1
    ).min(axis=1)
    c_group_mask  = min_dists > R_HIT_THRESHOLD
    print(f"전체 C-group: {c_group_mask.sum()} / {len(ids)} ({c_group_mask.mean():.1%})")

    fold_hits    = []
    all_states   = []
    oof_preds    = np.zeros((len(ids), 3), dtype=np.float32)
    oof_cg_masks = np.zeros(len(ids), dtype=bool)

    for fold in range(N_FOLDS):
        print(f"\n=== Fold {fold + 1}/{N_FOLDS} ===")
        hit, state, val_mask, val_preds, val_cg = train_fold(
            fold, ids, coords, labels, device
        )
        fold_hits.append(hit)
        all_states.append(state)
        oof_preds[val_mask]    = val_preds
        oof_cg_masks[val_mask] = val_cg
        print(f"Fold {fold + 1} best R-Hit: {hit:.4f}")

    print(f"\n{'='*50}")
    print(f"CV mean R-Hit: {np.mean(fold_hits):.4f} ± {np.std(fold_hits):.4f}")
    for i, h in enumerate(fold_hits):
        print(f"  Fold {i+1}: {h:.4f}")

    oof_hit   = r_hit(oof_preds, labels)
    c_oof_hit = r_hit(oof_preds[oof_cg_masks], labels[oof_cg_masks])
    ab_hit    = r_hit(oof_preds[~oof_cg_masks], labels[~oof_cg_masks])
    c_dist    = float(np.linalg.norm(
        oof_preds[oof_cg_masks] - labels[oof_cg_masks], axis=-1
    ).mean())

    print(f"\nOOF R-Hit (overall)    : {oof_hit:.4f}")
    print(f"OOF R-Hit (C-group)    : {c_oof_hit:.4f}  mean-dist: {c_dist*100:.2f}cm  "
          f"(samples: {oof_cg_masks.sum()})")
    print(f"OOF R-Hit (A+B-group)  : {ab_hit:.4f}  (samples: {(~oof_cg_masks).sum()})")

    # 모델 저장
    for i, state in enumerate(all_states):
        torch.save(state, out_dir / f"lstm_fold{i}.pt")
    np.save(out_dir / "lstm_oof_preds.npy", oof_preds)

    print(f"\nModels saved → {out_dir}/lstm_fold{{0..{N_FOLDS-1}}}.pt")
    print(f"OOF preds  → {out_dir}/lstm_oof_preds.npy")

    # 성공 기준 체크
    print(f"\n[성공 기준 체크]")
    c_pass = "PASS" if c_oof_hit >= 0.05 else "FAIL"
    print(f"  C-group R-Hit target  : 5.0%  [{c_pass}]  ({c_oof_hit*100:.2f}%)")
    print(f"  Overall OOF target    : 0.650 -> TBD after ensemble (LSTM single {oof_hit:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed (default: config.SEED)")
    args = parser.parse_args()
    train(seed=args.seed)
