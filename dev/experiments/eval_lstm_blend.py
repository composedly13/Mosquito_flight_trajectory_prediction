"""
로컬 OOF 기반 LSTM blend 평가 스크립트.

selector OOF: 각 샘플을 해당 fold 모델로 예측 (진짜 OOF)
lstm OOF:     seeds 42/777/123의 lstm_oof_preds.npy 평균
routing:      physics_routing_alpha (현재 regression.py 기본값 사용)

Usage:
    python dev/experiments/eval_lstm_blend.py
"""
import hashlib
import numpy as np
import torch
from tqdm import tqdm

from config import TRAIN_DIR, LABELS_PATH, OUTPUT_DIR, N_FOLDS, BATCH_SIZE, TOPK, R_HIT_THRESHOLD
from dataset import load_all
from model import CandidateSelector, selector_predict
from candidates import make_candidates, make_candidates_gpu, make_seq_features_gpu, make_cand_features_gpu
from regression import physics_routing_alpha


def r_hit(pred, true):
    return float(np.mean(np.linalg.norm(pred - true, axis=-1) <= R_HIT_THRESHOLD))


def fold_id(sample_id, n_folds=N_FOLDS):
    return int(hashlib.md5(sample_id.encode()).hexdigest()[:8], 16) % n_folds


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ids, coords, labels = load_all(TRAIN_DIR, LABELS_PATH)
    N = len(ids)
    print(f"Train samples: {N}")

    # C-group mask
    print("C-group mask 계산 중...")
    oracle = make_candidates(coords)
    min_dists = np.linalg.norm(oracle - labels[:, None], axis=-1).min(axis=1)
    c_mask = min_dists > R_HIT_THRESHOLD
    print(f"C-group: {c_mask.sum()} / {N} ({c_mask.mean():.1%})")

    # ── 1. Selector OOF (진짜 OOF: fold 모델로 해당 fold 샘플 예측) ──────────
    print("\n[Selector OOF 계산 중...]")
    fold_ids = np.array([fold_id(i) for i in ids])

    # 각 fold의 모델 로드
    fold_models = []
    for f in range(N_FOLDS):
        m = CandidateSelector().to(device)
        path = OUTPUT_DIR / f"selector_fold{f}.pt"
        m.load_state_dict(torch.load(path, map_location=device), strict=False)
        m.eval()
        fold_models.append(m)

    sel_oof = np.zeros((N, 3), dtype=np.float32)
    sel_seq  = np.zeros((N, 11, 11), dtype=np.float32)  # routing용 seq_features 저장

    for start in tqdm(range(0, N, BATCH_SIZE), desc="Selector OOF"):
        end  = min(start + BATCH_SIZE, N)
        c    = torch.tensor(coords[start:end]).to(device)
        fi   = fold_ids[start:end]

        with torch.no_grad():
            cands_t = make_candidates_gpu(c)
            seq_t   = make_seq_features_gpu(c)
            cand_t  = make_cand_features_gpu(c, cands_t)

        sel_seq[start:end] = seq_t.cpu().numpy()

        # fold별로 해당 샘플만 예측
        for f in range(N_FOLDS):
            mask = (fi == f)
            if mask.sum() == 0:
                continue
            idx = np.where(mask)[0]
            with torch.no_grad():
                logits = fold_models[f](seq_t[mask], cand_t[mask])
                pred   = selector_predict(logits, cands_t[mask], topk=TOPK, temp=2.0)
            sel_oof[start + idx] = pred.cpu().numpy()

    sel_hit   = r_hit(sel_oof, labels)
    sel_c_hit = r_hit(sel_oof[c_mask], labels[c_mask])
    print(f"Selector OOF R-Hit: {sel_hit:.4f}  (C-group: {sel_c_hit:.4f})")

    # ── 2. LSTM OOF (seeds 42/777/123 평균) ──────────────────────────────────
    print("\n[LSTM OOF 로드 중...]")
    lstm_seeds = [42, 777, 123]
    lstm_oofs  = []
    for s in lstm_seeds:
        p = OUTPUT_DIR / f"seed{s}" / "lstm_oof_preds.npy"
        if p.exists():
            lstm_oofs.append(np.load(p))
            print(f"  seed{s}: loaded")
        else:
            print(f"  seed{s}: 없음 (skip)")

    lstm_oof = np.mean(lstm_oofs, axis=0)  # (N, 3)
    lstm_hit   = r_hit(lstm_oof, labels)
    lstm_c_hit = r_hit(lstm_oof[c_mask], labels[c_mask])
    print(f"LSTM OOF R-Hit:     {lstm_hit:.4f}  (C-group: {lstm_c_hit:.4f})")

    # ── 3. Physics routing blend ──────────────────────────────────────────────
    print("\n[Physics routing blend 계산 중...]")
    seq_tensor = torch.tensor(sel_seq)  # CPU OK
    alpha = physics_routing_alpha(seq_tensor).numpy()[:, np.newaxis]  # (N, 1)

    n_routed = int((alpha > 0).sum())
    print(f"Routing 발동: {n_routed} / {N} ({n_routed/N*100:.1f}%)")
    print(f"alpha: mean={alpha.mean():.4f}  max={alpha.max():.3f}")

    blended = (1 - alpha) * sel_oof + alpha * lstm_oof

    blend_hit   = r_hit(blended, labels)
    blend_c_hit = r_hit(blended[c_mask], labels[c_mask])
    ab_mask     = ~c_mask

    print(f"\n{'='*50}")
    print(f"{'':20s}  {'전체':>8}  {'C-group':>8}  {'A+B-group':>10}")
    print(f"{'Selector OOF':20s}  {sel_hit:>8.4f}  {sel_c_hit:>8.4f}  {r_hit(sel_oof[ab_mask], labels[ab_mask]):>10.4f}")
    print(f"{'LSTM OOF':20s}  {lstm_hit:>8.4f}  {lstm_c_hit:>8.4f}  {r_hit(lstm_oof[ab_mask], labels[ab_mask]):>10.4f}")
    print(f"{'Blend OOF':20s}  {blend_hit:>8.4f}  {blend_c_hit:>8.4f}  {r_hit(blended[ab_mask], labels[ab_mask]):>10.4f}")
    delta = blend_hit - sel_hit
    print(f"\n  Blend vs Selector: {delta:+.4f} ({'개선' if delta > 0 else '악화'})")
    print(f"  routing 비율: {n_routed/N*100:.1f}%  (이전 13.4%)")


if __name__ == "__main__":
    main()
