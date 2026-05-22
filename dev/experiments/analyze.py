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
    make_candidates, motion_terms,
    make_candidates_gpu, make_seq_features_gpu, make_cand_features_gpu,
    N_CANDIDATES, CANDIDATES,
)


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
                  device: torch.device, topk: int = 3, temp: float = 1.0) -> np.ndarray:
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
                pred    = selector_predict(logits, cands_t, topk=topk, temp=temp)
            preds[val_idx[start:end]] = pred.cpu().numpy()
    return preds


def candidate_oracle_report(coords: np.ndarray, labels: np.ndarray) -> tuple:
    """Detailed oracle analysis: percentiles, top oracle candidate indices."""
    cands     = make_candidates(coords)                              # (N, C, 3)
    dist      = np.linalg.norm(cands - labels[:, np.newaxis, :], axis=-1)  # (N, C)
    best_dist = dist.min(axis=1)                                     # (N,)
    best_idx  = dist.argmin(axis=1)                                  # (N,)
    oracle_hit = float(np.mean(best_dist <= R_HIT_THRESHOLD))

    pcts = np.percentile(best_dist * 100, [50, 75, 90, 95])
    print(f"  Oracle R-Hit@1cm   : {oracle_hit:.4f}")
    print(f"  Best dist mean     : {best_dist.mean() * 100:.4f} cm")
    print(f"  Percentiles (cm)   : p50={pcts[0]:.3f}  p75={pcts[1]:.3f}"
          f"  p90={pcts[2]:.3f}  p95={pcts[3]:.3f}")

    uniq, cnts = np.unique(best_idx, return_counts=True)
    cnt_map = dict(zip(uniq.tolist(), cnts.tolist()))
    ranked = sorted(cnt_map.items(), key=lambda x: -x[1])

    print(f"\n  {'idx':>3}  {'name':<30}  {'count':>6}  {'%':>6}")
    print(f"  {'-'*3}  {'-'*30}  {'-'*6}  {'-'*6}")
    never_used = []
    for idx, cnt in ranked[:15]:
        pct = cnt / len(best_idx) * 100
        print(f"  {idx:>3}  {CANDIDATES[idx].name:<30}  {cnt:>6}  {pct:>5.1f}%")
    for i in range(N_CANDIDATES):
        if i not in cnt_map:
            never_used.append(i)
    if never_used:
        print(f"\n  미사용 후보 ({len(never_used)}개): "
              + ", ".join(f"{i}({CANDIDATES[i].name})" for i in never_used))

    return oracle_hit, best_dist, best_idx


def c_group_analysis(coords: np.ndarray, labels: np.ndarray) -> None:
    """
    C그룹 분석: 후보 공간에 없는 케이스의 Frenet 파라미터 분포.
    → 어떤 후보를 추가해야 oracle ceiling을 올릴 수 있는지 파악.
    """
    cands    = make_candidates(coords)                                       # (N, C, 3)
    dist_all = np.linalg.norm(cands - labels[:, np.newaxis, :], axis=-1)   # (N, C)
    best_dist = dist_all.min(axis=1)
    best_idx  = dist_all.argmin(axis=1)
    c_mask    = best_dist > R_HIT_THRESHOLD                                  # C그룹

    n_c = int(c_mask.sum())
    print(f"  C그룹: {n_c}개  ({c_mask.mean():.1%})")

    c_dists = best_dist[c_mask] * 100
    pcts = np.percentile(c_dists, [50, 75, 90, 95])
    print(f"  nearest-cand dist  : mean={c_dists.mean():.2f}cm  "
          f"p50={pcts[0]:.2f}  p75={pcts[1]:.2f}  p90={pcts[2]:.2f}  p95={pcts[3]:.2f}")

    # Frenet frame 분해
    p0, d1, d2, acc, jerk_vec = motion_terms(coords[c_mask], end_idx=10)
    speed      = np.linalg.norm(d1, axis=1, keepdims=True).clip(EPS)  # (N_c, 1)
    tangent    = d1 / speed
    acc_par_s  = (acc * tangent).sum(axis=1, keepdims=True)
    acc_perp_v = acc - acc_par_s * tangent
    perp_mag   = np.linalg.norm(acc_perp_v, axis=1, keepdims=True).clip(EPS)
    perp_unit  = acc_perp_v / perp_mag

    delta   = labels[c_mask] - p0                          # true label displacement
    t, t2   = 2.0, 4.0
    speed_h = speed[:, 0] * t                              # expected displacement scale

    # true label의 Frenet 투영 (후보 파라미터 스케일과 동일)
    par_norm  = (delta * tangent).sum(axis=1) / (speed_h + EPS)
    perp_norm = (delta * perp_unit).sum(axis=1) / (speed_h + EPS)
    jerk_norm_c = np.linalg.norm(jerk_vec, axis=1) / (speed[:, 0] + EPS)

    print(f"\n  True label (Frenet, speed×2 정규화):")
    print(f"  par  : mean={par_norm.mean():.2f}  std={par_norm.std():.2f}  "
          f"p5={np.percentile(par_norm,5):.2f}  p95={np.percentile(par_norm,95):.2f}")
    print(f"  perp : mean={perp_norm.mean():.2f}  std={perp_norm.std():.2f}  "
          f"|p5|={np.percentile(np.abs(perp_norm),5):.2f}  "
          f"|p75|={np.percentile(np.abs(perp_norm),75):.2f}  "
          f"|p95|={np.percentile(np.abs(perp_norm),95):.2f}")
    print(f"  jerk : mean={jerk_norm_c.mean():.2f}  "
          f"p75={np.percentile(jerk_norm_c,75):.2f}  p95={np.percentile(jerk_norm_c,95):.2f}")

    # 현재 후보 파라미터 커버리지와 비교
    cand_perp_max = max(abs(s.perp) for s in CANDIDATES)
    cand_par_max  = max(s.par for s in CANDIDATES)
    cand_jerk_max = max(abs(s.jerk) for s in CANDIDATES)
    print(f"\n  현재 후보 커버리지: par=[0, {cand_par_max:.2f}]  "
          f"|perp|≤{cand_perp_max:.2f}  |jerk|≤{cand_jerk_max:.2f}")

    exceed_perp = float(np.mean(np.abs(perp_norm) > cand_perp_max))
    exceed_par  = float(np.mean((par_norm > cand_par_max) | (par_norm < 0.3)))
    print(f"  C그룹 중 |perp| 초과: {exceed_perp:.1%}")
    print(f"  C그룹 중 par 범위 밖: {exceed_par:.1%}")

    # C그룹 가장 가까운 후보 top-10
    print(f"\n  C그룹 nearest 후보 분포 (top-10):")
    uniq, cnts = np.unique(best_idx[c_mask], return_counts=True)
    for idx, cnt in sorted(zip(uniq.tolist(), cnts.tolist()), key=lambda x: -x[1])[:10]:
        print(f"    [{idx:>2}] {CANDIDATES[idx].name:<30}  {cnt:>4}  ({cnt/n_c:.1%})")


def oracle_selector_decomposition(
    models: dict, ids: list, coords: np.ndarray, labels: np.ndarray,
    device: torch.device,
) -> None:
    """
    Decomposes selector error into three groups:
      A: oracle candidate exists AND is in top-5  (averaging may hurt)
      B: oracle candidate exists but NOT in top-5 (selector ranking problem)
      C: oracle candidate itself misses threshold  (candidate generation ceiling)
    """
    cands_np  = make_candidates(coords)                                       # (N, C, 3)
    dist_c    = np.linalg.norm(cands_np - labels[:, np.newaxis, :], axis=-1) # (N, C)
    oracle_idx  = dist_c.argmin(axis=1)            # (N,)
    oracle_dist = dist_c.min(axis=1)               # (N,)
    is_oracle_hit = oracle_dist <= R_HIT_THRESHOLD  # (N,) bool

    N = len(ids)
    all_logits = np.full((N, N_CANDIDATES), -np.inf, dtype=np.float32)
    for fold_idx, model in models.items():
        model.eval()
        val_mask = np.array([fold_id(i) == fold_idx for i in ids])
        val_arr  = coords[val_mask]
        val_idx  = np.where(val_mask)[0]
        for start in range(0, len(val_arr), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(val_arr))
            c = torch.tensor(val_arr[start:end]).to(device)
            with torch.no_grad():
                cands_t = make_candidates_gpu(c)
                seq_t   = make_seq_features_gpu(c)
                cand_t  = make_cand_features_gpu(c, cands_t)
                logits  = model(seq_t, cand_t)
            all_logits[val_idx[start:end]] = logits.cpu().numpy()

    # Oracle rank (0-based: 0 = top-1)
    sorted_idx  = np.argsort(-all_logits, axis=1)            # (N, C)
    oracle_rank = (sorted_idx == oracle_idx[:, np.newaxis]).argmax(axis=1)  # (N,)

    print(f"  Oracle candidate rank statistics:")
    print(f"    mean={oracle_rank.mean():.1f}  median={np.median(oracle_rank):.0f}"
          f"  p75={np.percentile(oracle_rank, 75):.0f}  p90={np.percentile(oracle_rank, 90):.0f}")
    for k in [1, 3, 5, 7]:
        rate = float(np.mean(oracle_rank < k))
        print(f"  Oracle in Top-{k}: {rate:.4f}")

    # Softmax weights (numerically stable)
    logits_shifted = all_logits - all_logits.max(axis=1, keepdims=True)
    exp_l   = np.exp(logits_shifted)
    weights = exp_l / exp_l.sum(axis=1, keepdims=True)                       # (N, C)

    def topk_pred(mask, k):
        idx_g = np.argsort(-all_logits[mask], axis=1)[:, :k]
        n_g   = mask.sum()
        w_g   = weights[mask][np.arange(n_g)[:, None], idx_g]
        w_g   = w_g / w_g.sum(axis=1, keepdims=True)
        c_g   = cands_np[mask][np.arange(n_g)[:, None], idx_g]
        return (c_g * w_g[:, :, np.newaxis]).sum(axis=1)

    oracle_in_top5 = oracle_rank < 5
    groups = {
        "A: oracle hit & in top-5   ": is_oracle_hit  & oracle_in_top5,
        "B: oracle hit & NOT top-5  ": is_oracle_hit  & ~oracle_in_top5,
        "C: oracle miss (cand limit)": ~is_oracle_hit,
    }
    print()
    for name, mask in groups.items():
        n = int(mask.sum())
        if n == 0:
            print(f"  {name}: 0 샘플")
            continue
        h1 = float(np.mean(np.linalg.norm(topk_pred(mask, 1) - labels[mask], axis=-1) <= R_HIT_THRESHOLD))
        h5 = float(np.mean(np.linalg.norm(topk_pred(mask, 5) - labels[mask], axis=-1) <= R_HIT_THRESHOLD))
        print(f"  {name}: {n:5d}샘플 ({mask.mean():.1%}) | Top-1 hit={h1:.4f}  Top-5 hit={h5:.4f}")

    print()
    print("  해석 가이드:")
    n_b = int((is_oracle_hit & ~oracle_in_top5).sum())
    n_a = int((is_oracle_hit & oracle_in_top5).sum())
    if n_b > n_a * 0.3:
        print("  → B그룹 비중 높음: selector가 oracle 후보를 top-5 밖으로 밀어냄 → 학습/손실 개선 필요")
    else:
        print("  → B그룹 비중 낮음: selector ranking은 양호")
    # Check if top-1 > top-5 in group A (averaging hurts)
    h1_a = float(np.mean(np.linalg.norm(
        topk_pred(is_oracle_hit & oracle_in_top5, 1) - labels[is_oracle_hit & oracle_in_top5],
        axis=-1) <= R_HIT_THRESHOLD)) if n_a > 0 else 0
    h5_a = float(np.mean(np.linalg.norm(
        topk_pred(is_oracle_hit & oracle_in_top5, 5) - labels[is_oracle_hit & oracle_in_top5],
        axis=-1) <= R_HIT_THRESHOLD)) if n_a > 0 else 0
    if h1_a > h5_a + 0.01:
        print("  → A그룹: Top-1 > Top-5 → 가중 평균이 성능을 깎음 → topk=1 검토")
    elif h5_a > h1_a + 0.01:
        print("  → A그룹: Top-5 > Top-1 → 가중 평균이 유효 → topk 유지")
    else:
        print("  → A그룹: Top-1 ≈ Top-5 → 가중 평균 영향 미미")


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
    phys_hit = r_hit(physics_pred(coords), labels)
    print(f"Physics baseline (β=0.6)              : {phys_hit:.4f}")
    oracle, min_dist, _ = candidate_oracle_report(coords, labels)
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
    for k in [1, 2, 3, 5, 7, 10, N_CANDIDATES]:
        p = get_oof_preds(models, ids, coords, device, topk=k)
        topk_preds[k] = p
        label = f"Top-{k}" if k < N_CANDIDATES else f"Top-all({N_CANDIDATES})"
        print(f"  {label}: {r_hit(p, labels):.4f}")

    best_k = max(topk_preds, key=lambda k: r_hit(topk_preds[k], labels))
    print(f"\n  → 최적 k: {best_k}")

    # ── 2b. Prediction temperature 비교 ───────────────────────────
    print("\n" + "=" * 55)
    print(f"2b. PREDICTION TEMPERATURE  (Top-{best_k} OOF)")
    print("=" * 55)
    best_temp, best_temp_hit = 1.0, 0.0
    temp_preds = {}
    for t in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        p = get_oof_preds(models, ids, coords, device, topk=best_k, temp=t)
        h = r_hit(p, labels)
        temp_preds[t] = p
        marker = ""
        if h > best_temp_hit:
            best_temp_hit, best_temp = h, t
            marker = "  ← best"
        print(f"  temp={t:.1f}: {h:.4f}{marker}")
    print(f"\n  → 최적 temperature: {best_temp}")

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

    # ── 4. Selector error decomposition ──────────────────────────
    print("\n" + "=" * 55)
    print("4. SELECTOR ERROR DECOMPOSITION")
    print("=" * 55)
    oracle_selector_decomposition(models, ids, coords, labels, device)

    # ── 4b. C그룹 분석 ───────────────────────────────────────────
    print("\n" + "=" * 55)
    print("4b. C그룹 분석  (후보 공간 갭 파악)")
    print("=" * 55)
    c_group_analysis(coords, labels)

    # ── 5. TTA 효과 ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("5. TTA  (Test-Time Augmentation, OOF)")
    print("=" * 55)
    print("  계산 중... (fold × n_tta 배치 실행)")
    tta_preds = tta_oof_preds(models, ids, coords, device, n_tta=8, topk=best_k)
    print(f"  Top-{best_k} no TTA  : {r_hit(topk_preds[best_k], labels):.4f}")
    print(f"  Top-{best_k} TTA×8   : {r_hit(tta_preds, labels):.4f}")

    # ── 6. 요약 ───────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("6. 요약")
    print("=" * 55)
    best_oof   = topk_preds[best_k]
    best_hit   = r_hit(best_oof, labels)
    selector_hit = r_hit(topk_preds.get(3, topk_preds[best_k]), labels)
    print(f"  Physics baseline       : {phys_hit:.4f}")
    print(f"  Oracle ceiling         : {oracle:.4f}")
    print(f"  Selector (Top-{best_k}, OOF)  : {best_hit:.4f}")
    print(f"  Selector efficiency    : {best_hit/oracle:.1%} of oracle")
    print(f"  Best blend (α={best_alpha:.1f})      : {best_blend:.4f}")
    print(f"  TTA×8                  : {r_hit(tta_preds, labels):.4f}")
    gap = oracle - best_hit
    print(f"\n  oracle↔selector 갭    : {gap:.4f}  ({gap*100:.1f}pp)")
    print(f"  70% 달성 조건          : efficiency ≥ {0.70/oracle:.1%}")


if __name__ == "__main__":
    analyze()
