"""
Post-hoc diagnostics using saved fold models.
Run: python dev/experiments/analyze.py
"""
import sys, hashlib
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from config import *
from dataset import load_all, augment_batch_gpu_with_R
from model import CandidateSelector, selector_predict
from candidates import (
    make_candidates,
    make_candidates_gpu, make_seq_features_gpu, make_cand_features_gpu,
    N_CANDIDATES,
)
from boundary import BoundaryMLP, apply_boundary


def r_hit(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(pred - true, axis=-1) <= R_HIT_THRESHOLD))


def fold_id(sample_id: str, n_folds: int = N_FOLDS) -> int:
    return int(hashlib.md5(sample_id.encode()).hexdigest()[:8], 16) % n_folds


def physics_pred(coords: np.ndarray) -> np.ndarray:
    """Linear + acceleration extrapolation (β=0.6)."""
    d1 = coords[:, -1] - coords[:, -2]
    d2 = coords[:, -2] - coords[:, -3]
    return coords[:, -1] + 2 * d1 + 0.6 * (d1 - d2)


def tta_oof_preds(models: dict, ids: list, coords: np.ndarray,
                  device: torch.device, n_tta: int = 8, topk: int = 3) -> np.ndarray:
    """OOF predictions with TTA: average N random SO3 rotations."""
    N = len(ids)
    pred_sum = np.zeros((N, 3), dtype=np.float32)
    for fold_idx, model in models.items():
        model.eval()
        val_mask = np.array([fold_id(i) == fold_idx for i in ids])
        val_coords = coords[val_mask]
        val_idx    = np.where(val_mask)[0]
        for start in range(0, len(val_coords), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(val_coords))
            c   = torch.tensor(val_coords[start:end]).to(device)
            fold_pred = torch.zeros(end - start, 3, device=device)
            with torch.no_grad():
                for aug_i in range(n_tta):
                    if aug_i == 0:
                        c_in, R = c, torch.eye(3, device=device).unsqueeze(0).expand(len(c), -1, -1)
                    else:
                        c_in, R = augment_batch_gpu_with_R(c)
                    cands_t = make_candidates_gpu(c_in)
                    seq_t   = make_seq_features_gpu(c_in)
                    cand_t  = make_cand_features_gpu(c_in, cands_t)
                    logits  = model(seq_t, cand_t)
                    pred_r  = selector_predict(logits, cands_t, topk=topk)
                    # Unrotate back to original frame: pred_orig = pred_rot @ R
                    pred_orig = (pred_r.unsqueeze(1) @ R).squeeze(1)
                    fold_pred += pred_orig
            pred_sum[val_idx[start:end]] = (fold_pred / n_tta).cpu().numpy()
    return pred_sum


def get_oof_preds(models: dict, ids: list, coords: np.ndarray,
                  device: torch.device, topk: int = 3) -> np.ndarray:
    N = len(ids)
    preds = np.zeros((N, 3), dtype=np.float32)
    for fold_idx, model in models.items():
        model.eval()
        val_mask = np.array([fold_id(i) == fold_idx for i in ids])
        val_coords = coords[val_mask]
        val_idx    = np.where(val_mask)[0]
        for start in range(0, len(val_coords), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(val_coords))
            c = torch.tensor(val_coords[start:end]).to(device)
            with torch.no_grad():
                cands_t = make_candidates_gpu(c)
                seq_t   = make_seq_features_gpu(c)
                cand_t  = make_cand_features_gpu(c, cands_t)
                logits  = model(seq_t, cand_t)
                pred    = selector_predict(logits, cands_t, topk=topk)
            preds[val_idx[start:end]] = pred.cpu().numpy()
    return preds


def analyze():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids, coords, labels = load_all(TRAIN_DIR, LABELS_PATH)

    # Load saved fold models
    models = {}
    for fold in range(N_FOLDS):
        path = OUTPUT_DIR / f"selector_fold{fold}.pt"
        if path.exists():
            m = CandidateSelector().to(device)
            m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            m.eval()
            models[fold] = m
    if not models:
        print("저장된 모델 없음. train.py 먼저 실행.")
        return
    print(f"모델 {len(models)}개 로드 완료\n")

    # ── 1. Oracle hit ──────────────────────────────────────────────
    print("=" * 55)
    print(f"1. ORACLE HIT  (후보군 상한선, N={N_CANDIDATES})")
    print("=" * 55)
    oracle_cands = make_candidates(coords)                    # (N, C, 3)
    min_dist = np.linalg.norm(
        oracle_cands - labels[:, np.newaxis, :], axis=-1
    ).min(axis=1)                                             # (N,)
    oracle   = float(np.mean(min_dist <= R_HIT_THRESHOLD))
    phys_hit = r_hit(physics_pred(coords), labels)

    print(f"Physics baseline (β=0.6)              : {phys_hit:.4f}")
    print(f"Oracle R-Hit (best of {N_CANDIDATES} candidates) : {oracle:.4f}")
    print()
    if oracle < 0.68:
        print("→ 후보군 자체가 병목. 후보 수/파라미터 확장 필요.")
    elif oracle < 0.73:
        print("→ Selector 품질이 주 병목. 모델/손실 개선 여지 있음.")
    else:
        print("→ 후보군 충분. 학습·선택 방식 최적화 우선.")

    # ── 2. Top-k 비교 ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("2. TOP-K 비교  (OOF)")
    print("=" * 55)
    topk_preds = {}
    for k in [1, 2, 3, 5]:
        p = get_oof_preds(models, ids, coords, device, topk=k)
        topk_preds[k] = p
        print(f"  Top-{k}: {r_hit(p, labels):.4f}")

    best_k = max(topk_preds, key=lambda k: r_hit(topk_preds[k], labels))
    print(f"\n  → 최적 k: {best_k}")

    # ── 3. Physics blend ──────────────────────────────────────────
    print("\n" + "=" * 55)
    print("3. PHYSICS BLEND  (Top-3 OOF, α=모델 비중)")
    print("=" * 55)
    phys = physics_pred(coords)
    p3   = topk_preds[3]
    best_alpha, best_blend = 1.0, r_hit(p3, labels)
    for alpha in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        blended = alpha * p3 + (1 - alpha) * phys
        h = r_hit(blended, labels)
        marker = "  ← best" if h > best_blend else ""
        if h > best_blend:
            best_blend, best_alpha = h, alpha
        print(f"  α={alpha:.1f}: {h:.4f}{marker}")

    # ── 4. TTA 효과 ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("4. TTA  (Test-Time Augmentation, OOF)")
    print("=" * 55)
    print("  계산 중... (fold × n_tta 배치 실행)")
    tta_preds = tta_oof_preds(models, ids, coords, device, n_tta=8, topk=best_k)
    print(f"  Top-{best_k} no TTA  : {r_hit(topk_preds[best_k], labels):.4f}")
    print(f"  Top-{best_k} TTA×8   : {r_hit(tta_preds, labels):.4f}")

    # ── 6. Boundary MLP ───────────────────────────────────────────
    print("\n" + "=" * 55)
    print("5. BOUNDARY MLP 효과")
    print("=" * 55)
    bpath = OUTPUT_DIR / "boundary.pt"
    if bpath.exists():
        bm = BoundaryMLP().to(device)
        bm.load_state_dict(torch.load(bpath, map_location=device, weights_only=True))
        corrected = apply_boundary(bm, coords, p3, device)
        print(f"  Top-3 selector only : {r_hit(p3, labels):.4f}")
        print(f"  Top-3 + Boundary    : {r_hit(corrected, labels):.4f}")
        diff = r_hit(corrected, labels) - r_hit(p3, labels)
        print(f"  차이                : {diff:+.4f}")
        if diff < 0:
            print("\n  → Boundary MLP 해로움. 제출 시 boundary 제거 권장.")
    else:
        print("  boundary.pt 없음")

    # ── 7. 요약 ───────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("6. 요약")
    print("=" * 55)
    selector_hit = r_hit(p3, labels)
    print(f"  Physics baseline       : {phys_hit:.4f}")
    print(f"  Oracle ceiling         : {oracle:.4f}")
    print(f"  Selector (Top-3, OOF)  : {selector_hit:.4f}")
    print(f"  Selector efficiency    : {selector_hit/oracle:.1%} of oracle")
    print(f"  Best blend (α={best_alpha:.1f})      : {best_blend:.4f}")
    print(f"  TTA×8                  : {r_hit(tta_preds, labels):.4f}")
    gap = oracle - selector_hit
    print(f"\n  oracle↔selector 갭    : {gap:.4f}  ({gap*100:.1f}pp)")
    print(f"  (oracle은 {N_CANDIDATES}개 후보 기준, 재학습 후 상승 가능)")


if __name__ == "__main__":
    analyze()
