"""
C-group residual correction 학습.
C-labeled 샘플에서 base_pred + delta → y_true 학습.

핵심 개선: base_pred - p0 (base 방향 벡터)를 입력에 추가
  → GRU가 "평균 delta" 학습 문제 해결, per-sample 방향 보정 가능

입력: export_oof_phase13.py 저장값 (OOF base_pred, seq_feat, y_true, c_labels)
출력:
  outputs/residual_c/residual_fold{i}.pt
  outputs/residual_c/residual_oof_delta.npy  (N, 3) -- C samples only, 0 elsewhere

Usage:
    python train_residual_c.py [--threshold 0.70] [--alpha 0.5]
"""
import argparse, hashlib, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from dataset import load_all

OOF_DIR    = OUTPUT_DIR / "oof_phase13"
GATE_DIR   = OUTPUT_DIR / "c_gate"
RESID_DIR  = OUTPUT_DIR / "residual_c"

CORRECTION_CAP = 0.006   # 6mm, config.py의 CORRECTION_CAP과 동일
DELTA_LAMBDA   = 0.05    # delta norm regularization weight


class GRUResidual(nn.Module):
    """
    C-group 전용 잔차 보정 모델.
    final_pred = base_pred + alpha * clamp(delta, -CAP, CAP)

    base_delta = base_pred - p0: base 방향 정보 주입
    → per-sample 방향 보정 가능, 평균 수렴 문제 해소
    """
    def __init__(self, seq_dim: int = 11, hidden: int = 64, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=seq_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        # GRU hidden + base_delta(3) → delta(3)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + 3, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, seq_feat: torch.Tensor, base_delta: torch.Tensor) -> torch.Tensor:
        """
        seq_feat  : (B, 11, 11)
        base_delta: (B, 3)  = base_pred - p0  (base 방향 벡터)
        → delta   : (B, 3) clamped to ±CORRECTION_CAP
        """
        _, h_n = self.gru(seq_feat)
        h      = torch.cat([h_n[-2], h_n[-1]], dim=-1)   # (B, hidden*2)
        h_cat  = torch.cat([h, base_delta], dim=-1)       # (B, hidden*2 + 3)
        delta  = self.head(h_cat)
        return delta.clamp(-CORRECTION_CAP, CORRECTION_CAP)


def fold_id(sample_id, n_folds=N_FOLDS):
    return int(hashlib.md5(sample_id.encode()).hexdigest()[:8], 16) % n_folds


def r_hit(pred, true):
    return float(np.mean(np.linalg.norm(pred - true, axis=-1) <= R_HIT_THRESHOLD))


def eval_blend(base_preds, delta_oof, c_prob_oof, y_true, threshold, alpha, c_labels,
               oracle_min_dist):
    pred_c = c_prob_oof >= threshold
    final  = base_preds.copy()
    final[pred_c] = base_preds[pred_c] + alpha * delta_oof[pred_c]

    near_c = (oracle_min_dist > R_HIT_THRESHOLD) & (oracle_min_dist <= 0.015)
    hard_c = oracle_min_dist > 0.015

    base_all  = r_hit(base_preds, y_true)
    final_all = r_hit(final, y_true)
    pred_c_n  = int(pred_c.sum())
    tp        = int((pred_c & c_labels).sum())

    print(f"  alpha={alpha:.2f}  threshold={threshold:.2f}  pred_C={pred_c_n}({pred_c.mean():.1%})")
    print(f"  predicted C: prec={tp/max(pred_c_n,1):.4f}  rec={tp/c_labels.sum():.4f}")
    print(f"  base  R-Hit (all)   : {base_all:.4f}")
    print(f"  final R-Hit (all)   : {final_all:.4f}  ({(final_all-base_all)*100:+.2f}pp)")
    if pred_c_n > 0:
        b_c = r_hit(base_preds[pred_c], y_true[pred_c])
        f_c = r_hit(final[pred_c],      y_true[pred_c])
        print(f"  pred-C subset: base={b_c:.4f}  final={f_c:.4f}  ({(f_c-b_c)*100:+.2f}pp)")
    nc_mask = ~pred_c
    if nc_mask.sum() > 0:
        b_nc = r_hit(base_preds[nc_mask], y_true[nc_mask])
        f_nc = r_hit(final[nc_mask],      y_true[nc_mask])
        print(f"  non-C subset: base={b_nc:.4f}  final={f_nc:.4f}  ({(f_nc-b_nc)*100:+.2f}pp)")
    if near_c.sum() > 0:
        fn = r_hit(final[near_c], y_true[near_c])
        bn = r_hit(base_preds[near_c], y_true[near_c])
        print(f"  near-C  (1~1.5cm): base={bn:.4f}  final={fn:.4f}  ({(fn-bn)*100:+.2f}pp)")
    if hard_c.sum() > 0:
        fh = r_hit(final[hard_c], y_true[hard_c])
        bh = r_hit(base_preds[hard_c], y_true[hard_c])
        print(f"  hard-C  (>1.5cm) : base={bh:.4f}  final={fh:.4f}  ({(fh-bh)*100:+.2f}pp)")
    return final_all


def main(threshold: float = 0.70, alpha: float = 0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  threshold={threshold}  |  alpha={alpha}")

    # ── 데이터 로드 ───────────────────────────────────────────────
    for fname in ["oof_preds.npy", "oof_seq_feat.npy", "c_labels.npy",
                  "oracle_min_dist.npy", "y_true.npy", "sample_ids.npy"]:
        if not (OOF_DIR / fname).exists():
            raise FileNotFoundError(f"Missing: {OOF_DIR / fname}\n"
                                    f"  먼저: python export_oof_phase13.py")

    if not (GATE_DIR / "c_gate_oof_probs.npy").exists():
        raise FileNotFoundError(f"Missing: {GATE_DIR}/c_gate_oof_probs.npy\n"
                                f"  먼저: python train_c_gate.py")

    base_preds      = np.load(OOF_DIR / "oof_preds.npy")        # (N, 3)
    oof_seq_feat    = np.load(OOF_DIR / "oof_seq_feat.npy")      # (N, 11, 11)
    c_labels        = np.load(OOF_DIR / "c_labels.npy")          # (N,) bool
    oracle_min_dist = np.load(OOF_DIR / "oracle_min_dist.npy")   # (N,)
    y_true          = np.load(OOF_DIR / "y_true.npy")            # (N, 3)
    sample_ids      = np.load(OOF_DIR / "sample_ids.npy")        # (N,) str
    c_prob_oof      = np.load(GATE_DIR / "c_gate_oof_probs.npy") # (N,)

    # p0: 마지막 알려진 위치 (coords[:, 10, :])
    _, coords, _ = load_all(TRAIN_DIR, LABELS_PATH)
    p0_all = coords[:, 10, :]                                    # (N, 3)

    # base_delta: base 방향 벡터 (핵심 feature)
    base_delta_all = base_preds - p0_all                         # (N, 3)

    N = len(base_preds)
    fold_ids = np.array([fold_id(i) for i in sample_ids])

    n_c = c_labels.sum()
    print(f"N={N}  C={n_c}({n_c/N:.1%})  "
          f"near-C={((oracle_min_dist>R_HIT_THRESHOLD)&(oracle_min_dist<=0.015)).sum()}  "
          f"hard-C={(oracle_min_dist>0.015).sum()}")

    RESID_DIR.mkdir(parents=True, exist_ok=True)

    # ── 5-fold residual 학습 (C samples only) ─────────────────────
    delta_oof = np.zeros((N, 3), dtype=np.float32)

    for fold in range(N_FOLDS):
        val_fold_mask = fold_ids == fold
        trn_fold_mask = ~val_fold_mask

        trn_c_mask = trn_fold_mask & c_labels
        val_c_mask = val_fold_mask & c_labels

        if trn_c_mask.sum() == 0 or val_c_mask.sum() == 0:
            print(f"  Fold {fold}: C samples 부족 -- skip")
            continue

        X_trn  = torch.tensor(oof_seq_feat[trn_c_mask]).to(device)   # (Nc_trn, 11, 11)
        bd_trn = torch.tensor(base_delta_all[trn_c_mask]).to(device)  # (Nc_trn, 3)
        b_trn  = torch.tensor(base_preds[trn_c_mask]).to(device)      # (Nc_trn, 3)
        y_trn  = torch.tensor(y_true[trn_c_mask]).to(device)          # (Nc_trn, 3)

        X_val  = torch.tensor(oof_seq_feat[val_c_mask]).to(device)
        bd_val = torch.tensor(base_delta_all[val_c_mask]).to(device)
        b_val  = base_preds[val_c_mask]
        y_val  = y_true[val_c_mask]

        model     = GRUResidual().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

        best_hit, best_state = 0.0, None
        patience_cnt = 0
        delta_oof_val = np.zeros((val_c_mask.sum(), 3), dtype=np.float32)

        pbar = tqdm(range(300), desc=f"Fold {fold} C={trn_c_mask.sum()}", ncols=100)
        for epoch in pbar:
            model.train()
            perm = torch.randperm(len(X_trn))
            for start in range(0, len(X_trn), BATCH_SIZE):
                idx   = perm[start:start + BATCH_SIZE]
                delta = model(X_trn[idx], bd_trn[idx])
                pred  = b_trn[idx] + delta
                loss  = F.smooth_l1_loss(pred / R_HIT_THRESHOLD,
                                         y_trn[idx] / R_HIT_THRESHOLD,
                                         beta=1.0)
                loss  = loss + DELTA_LAMBDA * delta.norm(dim=-1).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                delta_v = model(X_val, bd_val).cpu().numpy()
            pred_v = b_val + delta_v
            hit_v  = r_hit(pred_v, y_val)
            base_v = r_hit(b_val, y_val)

            pbar.set_postfix(hit=f"{hit_v:.4f}", base=f"{base_v:.4f}", patience=patience_cnt)

            if hit_v > best_hit:
                best_hit      = hit_v
                best_state    = {k: v.clone() for k, v in model.state_dict().items()}
                delta_oof_val = delta_v.copy()
                patience_cnt  = 0
            else:
                patience_cnt += 1
                if patience_cnt >= 50:
                    break

        model.load_state_dict(best_state)
        torch.save(model.state_dict(), RESID_DIR / f"residual_fold{fold}.pt")

        val_c_idx = np.where(val_c_mask)[0]
        delta_oof[val_c_idx] = delta_oof_val

        print(f"  Fold {fold}: C val={val_c_mask.sum()}  "
              f"base={r_hit(b_val, y_val):.4f}  best={best_hit:.4f}  "
              f"({(best_hit - r_hit(b_val, y_val))*100:+.2f}pp)")

    np.save(RESID_DIR / "residual_oof_delta.npy", delta_oof)

    # ── 최종 OOF 평가 ────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"[최종 OOF 블렌딩 평가]")
    print(f"{'='*55}")

    best_final, best_alpha = 0.0, alpha

    for a in [0.3, 0.5, 0.7, 1.0]:
        print(f"\n--- alpha={a:.1f} ---")
        final_hit = eval_blend(
            base_preds, delta_oof, c_prob_oof, y_true,
            threshold, a, c_labels, oracle_min_dist,
        )
        if final_hit > best_final:
            best_final = final_hit
            best_alpha = a

    base_all = r_hit(base_preds, y_true)
    print(f"\n{'='*55}")
    print(f"[요약]")
    print(f"  base OOF R-Hit   : {base_all:.4f}")
    print(f"  best final R-Hit : {best_final:.4f}  (alpha={best_alpha:.1f})")
    print(f"  개선폭            : {(best_final - base_all)*100:+.2f}pp")

    if best_final <= base_all:
        print(f"\n[판정] OOF 개선 없음 -- Phase 13 C-gate 보류")
    elif best_final - base_all < 0.001:
        print(f"\n[판정] 개선 미미 (<0.1pp) -- 추가 튜닝 필요")
    else:
        print(f"\n[판정] OOF 개선 확인 -- predict_phase13.py 제출 진행")
        print(f"  python predict_phase13.py --threshold {threshold:.2f} --alpha {best_alpha:.1f}")

    print(f"\nSaved:")
    print(f"  {RESID_DIR}/residual_fold*.pt")
    print(f"  {RESID_DIR}/residual_oof_delta.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--alpha",     type=float, default=0.50)
    args = parser.parse_args()
    main(args.threshold, args.alpha)
