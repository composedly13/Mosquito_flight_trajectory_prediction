"""
Post-hoc diagnostics using saved fold models.
Run: python dev/experiments/analyze.py [--seed 42]
"""
import sys, hashlib, argparse
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
from regression import RegMLP, entropy_blend, _make_features as _reg_features, _fold_id as _reg_fold_id


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


def get_oof_logits(models: dict, ids: list, coords: np.ndarray,
                   device: torch.device) -> np.ndarray:
    """Return OOF raw selector logits (N, C)."""
    N      = len(ids)
    logits = np.full((N, N_CANDIDATES), -np.inf, dtype=np.float32)
    for fold_idx, model in models.items():
        model.eval()
        val_mask = np.array([fold_id(i) == fold_idx for i in ids])
        val_arr  = coords[val_mask]
        val_pos  = np.where(val_mask)[0]
        for start in range(0, len(val_arr), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(val_arr))
            c = torch.tensor(val_arr[start:end]).to(device)
            with torch.no_grad():
                cands_t = make_candidates_gpu(c)
                seq_t   = make_seq_features_gpu(c)
                cand_t  = make_cand_features_gpu(c, cands_t)
                lgts    = model(seq_t, cand_t)
            logits[val_pos[start:end]] = lgts.cpu().numpy()
    return logits


def load_all_models(seeds: list, device: torch.device, out_tag: str = "") -> dict:
    """Load fold models for multiple seeds. Returns {seed: {fold_idx: model}}."""
    all_models = {}
    for seed in seeds:
        model_dir = OUTPUT_DIR / f"seed{seed}{out_tag}"
        fold_models = {}
        for fold in range(N_FOLDS):
            path = model_dir / f"selector_fold{fold}.pt"
            if path.exists():
                m = CandidateSelector().to(device)
                m.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=False)
                m.eval()
                fold_models[fold] = m
        if fold_models:
            all_models[seed] = fold_models
            print(f"  seed{seed}{out_tag}: {len(fold_models)}/{N_FOLDS} folds 로드")
        else:
            print(f"  seed{seed}{out_tag}: 모델 없음 (python train.py --seed {seed} --out-tag '{out_tag}' 먼저 실행)")
    return all_models


def get_oof_preds_multiseed(
    all_models: dict,
    ids: list,
    coords: np.ndarray,
    device: torch.device,
    topk: int = 10,
    temp: float = 1.0,
) -> np.ndarray:
    """Multi-seed OOF: for each fold, average logits across seeds then predict."""
    N = len(ids)
    preds = np.zeros((N, 3), dtype=np.float32)
    seeds = list(all_models.keys())

    for fold_idx in range(N_FOLDS):
        val_mask = np.array([fold_id(i) == fold_idx for i in ids])
        if not val_mask.any():
            continue
        val_coords = coords[val_mask]
        val_pos    = np.where(val_mask)[0]

        fold_models = [all_models[s][fold_idx] for s in seeds if fold_idx in all_models[s]]
        if not fold_models:
            continue

        for start in range(0, len(val_coords), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(val_coords))
            c = torch.tensor(val_coords[start:end]).to(device)
            with torch.no_grad():
                cands_t    = make_candidates_gpu(c)
                seq_t      = make_seq_features_gpu(c)
                cand_t     = make_cand_features_gpu(c, cands_t)
                avg_logits = sum(m(seq_t, cand_t) for m in fold_models) / len(fold_models)
                pred       = selector_predict(avg_logits, cands_t, topk=topk, temp=temp)
            preds[val_pos[start:end]] = pred.cpu().numpy()

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
    for k in [1, 3, 5, 7, 10]:
        rate = float(np.mean(oracle_rank < k))
        print(f"  Oracle in Top-{k:>2}: {rate:.4f}")

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
        h1  = float(np.mean(np.linalg.norm(topk_pred(mask,  1) - labels[mask], axis=-1) <= R_HIT_THRESHOLD))
        h5  = float(np.mean(np.linalg.norm(topk_pred(mask,  5) - labels[mask], axis=-1) <= R_HIT_THRESHOLD))
        h10 = float(np.mean(np.linalg.norm(topk_pred(mask, 10) - labels[mask], axis=-1) <= R_HIT_THRESHOLD))
        print(f"  {name}: {n:5d}샘플 ({mask.mean():.1%}) | Top-1={h1:.4f}  Top-5={h5:.4f}  Top-10={h10:.4f}")

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


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 helpers — B-group diagnostic, confidence features, calibration
# ─────────────────────────────────────────────────────────────────────────────

def _topk_preds_np(logits: np.ndarray, cands_np: np.ndarray,
                   topk_arr: np.ndarray, temp: float = 1.0) -> np.ndarray:
    """Vectorized per-sample dynamic-topk prediction (numpy only).
    logits  : (N, C)
    cands_np: (N, C, 3)
    topk_arr: (N,) int — per-sample topk
    Returns : (N, 3)
    """
    N, C, _ = cands_np.shape
    probs = np.exp(logits / temp - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    sorted_idx   = np.argsort(-probs, axis=1)                              # (N, C)
    sorted_probs = probs[np.arange(N)[:, None], sorted_idx]               # (N, C)
    sorted_cands = cands_np[np.arange(N)[:, None], sorted_idx]            # (N, C, 3)
    mask  = np.arange(C)[None, :] < topk_arr[:, None]                     # (N, C)
    mw    = sorted_probs * mask
    mw   /= mw.sum(axis=1, keepdims=True) + 1e-9
    return (sorted_cands * mw[:, :, None]).sum(axis=1)                     # (N, 3)


def get_oof_logits_cands(models: dict, ids: list, coords: np.ndarray,
                         device: torch.device) -> tuple:
    """OOF logits + candidate positions (shared across sections 9-12).
    Returns: (logits (N,C), cands_np (N,C,3))
    """
    N = len(ids)
    logits_arr = np.full((N, N_CANDIDATES), -np.inf, dtype=np.float32)
    cands_np   = make_candidates(coords)                                   # (N, C, 3) — deterministic

    for fold_idx, model in models.items():
        model.eval()
        val_mask = np.array([fold_id(i) == fold_idx for i in ids])
        val_arr  = coords[val_mask]
        val_pos  = np.where(val_mask)[0]
        for start in range(0, len(val_arr), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(val_arr))
            c   = torch.tensor(val_arr[start:end]).to(device)
            with torch.no_grad():
                ct = make_candidates_gpu(c)
                st = make_seq_features_gpu(c)
                ft = make_cand_features_gpu(c, ct)
                lg = model(st, ft)
            logits_arr[val_pos[start:end]] = lg.cpu().numpy()
    return logits_arr, cands_np


def compute_conf_features(logits: np.ndarray, cands_np: np.ndarray,
                          coords: np.ndarray, temp: float = 1.0) -> dict:
    """Per-sample confidence features.
    Returns dict of (N,) arrays.
    """
    N, C = logits.shape
    probs = np.exp(logits / temp - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    sorted_logits = np.sort(logits, axis=1)[:, ::-1]
    sorted_probs  = np.sort(probs,  axis=1)[:, ::-1]

    # 1. top1-top2 logit gap
    gap = sorted_logits[:, 0] - sorted_logits[:, 1]

    # 2. top-5 entropy (normalized)
    p5  = sorted_probs[:, :5].copy()
    p5 /= p5.sum(axis=1, keepdims=True) + 1e-9
    ent5 = -(p5 * np.log(p5 + 1e-9)).sum(axis=1) / np.log(5)

    # 3. top-10 entropy (normalized)
    p10  = sorted_probs[:, :10].copy()
    p10 /= p10.sum(axis=1, keepdims=True) + 1e-9
    ent10 = -(p10 * np.log(p10 + 1e-9)).sum(axis=1) / np.log(10)

    # 4. candidate spread — weighted std of top-10 positions (cm)
    idx10 = np.argsort(-probs, axis=1)[:, :10]
    tc    = cands_np[np.arange(N)[:, None], idx10]              # (N, 10, 3)
    tw    = probs[np.arange(N)[:, None], idx10]
    tw   /= tw.sum(axis=1, keepdims=True) + 1e-9
    mc    = (tc * tw[:, :, None]).sum(axis=1)                    # (N, 3)
    spread_cm = 100 * np.sqrt(
        ((tc - mc[:, None]) ** 2 * tw[:, :, None]).sum(axis=(1, 2)) + 1e-8
    )

    # 5. physics disagreement (cm)
    d1v  = coords[:, -1] - coords[:, -2]
    d2v  = coords[:, -2] - coords[:, -3]
    phys = coords[:, -1] + 2 * d1v + 0.6 * (d1v - d2v)
    sel_top10 = _topk_preds_np(logits, cands_np, np.full(N, 10, np.int64), temp)
    phys_dis  = 100 * np.linalg.norm(sel_top10 - phys, axis=1)

    # 6-7. obs_acc_perp_abs + obs_jerk_abs (normalized by speed)
    p0_, d1_, d2_, acc_, jk_ = motion_terms(coords, end_idx=10)
    sp_  = np.linalg.norm(d1_, axis=1, keepdims=True).clip(EPS)
    tg_  = d1_ / sp_
    aps_ = (acc_ * tg_).sum(axis=1, keepdims=True)
    apv_ = acc_ - aps_ * tg_
    acc_perp_abs = np.linalg.norm(apv_, axis=1) / (sp_[:, 0] + EPS)
    jerk_abs     = np.linalg.norm(jk_,  axis=1) / (sp_[:, 0] + EPS)

    return {
        "top1_top2_gap":    gap,
        "top5_entropy":     ent5,
        "top10_entropy":    ent10,
        "spread_cm":        spread_cm,
        "phys_dis_cm":      phys_dis,
        "obs_acc_perp_abs": acc_perp_abs,
        "obs_jerk_abs":     jerk_abs,
    }


def analyze_b_group(logits: np.ndarray, cands_np: np.ndarray,
                    coords: np.ndarray, labels: np.ndarray,
                    oracle_idx: np.ndarray, oracle_rank: np.ndarray,
                    oracle_dist: np.ndarray) -> tuple:
    """
    B-group diagnostic.
    Prints: oracle rank distribution, family comparison, feature comparison.
    Returns: (b_mask (N,), par_match_b (N_b, C) or None)
    """
    from candidates import (make_cand_features as _mcf, CANDIDATE_FAMILY,
                             FAMILY_NAMES, CAND_FEAT_INTERACTION)  # type: ignore

    is_hit = oracle_dist <= R_HIT_THRESHOLD
    in_t5  = oracle_rank < 5
    b_mask = is_hit & ~in_t5
    n_b    = int(b_mask.sum())
    print(f"  B-group: {n_b}  ({b_mask.mean():.1%})")

    top1_idx_b   = logits[b_mask].argmax(axis=1)
    oracle_idx_b = oracle_idx[b_mask]
    rank_b       = oracle_rank[b_mask]

    # Oracle rank distribution
    print(f"\n  Oracle rank 분포 (B-group):")
    for k in [6, 7, 8, 10, 12, 15, 20, 30]:
        nk = int((rank_b < k).sum())
        print(f"    rank < {k:>2}: {nk:5d} ({nk/n_b:.1%})")

    # Family comparison
    t1f = CANDIDATE_FAMILY[top1_idx_b]
    orf = CANDIDATE_FAMILY[oracle_idx_b]
    print(f"\n  Family 분포 비교 (B-group):")
    print(f"  {'family':<10}  {'top1':>5}  {'top1%':>6}  {'oracle':>7}  {'orc%':>6}  {'over_sel':>8}")
    print(f"  {'-'*10}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*8}")
    for fid, fname in enumerate(FAMILY_NAMES):
        t1c = int((t1f == fid).sum())
        orc = int((orf == fid).sum())
        print(f"  {fname:<10}  {t1c:>5}  {t1c/n_b:>5.1%}  {orc:>7}  {orc/n_b:>5.1%}"
              f"  {(t1c-orc)/n_b:>+7.1%}")

    # Candidate feature comparison
    n_f = 14 if CAND_FEAT_INTERACTION else 10
    feat_names = [
        "cand_par", "cand_perp", "dist",
        "d1", "par", "perp", "d2", "jerk", "time_scale", "obs_acc_par",
        "obs_acc_perp", "par_match", "perp_match", "jerk_match",
    ][:n_f]
    cf_b = _mcf(coords[b_mask], cands_np[b_mask])               # (N_b, C, n_f)
    t1f_v  = cf_b[np.arange(n_b), top1_idx_b]                   # (N_b, n_f)
    orf_v  = cf_b[np.arange(n_b), oracle_idx_b]                 # (N_b, n_f)
    print(f"\n  Candidate feature 비교 (B-group top1 vs oracle):")
    print(f"  {'feature':<14}  {'top1':>9}  {'oracle':>9}  {'diff':>8}")
    print(f"  {'-'*14}  {'-'*9}  {'-'*9}  {'-'*8}")
    for i, fn in enumerate(feat_names):
        tm = float(t1f_v[:, i].mean()); om = float(orf_v[:, i].mean())
        print(f"  {fn:<14}  {tm:>9.4f}  {om:>9.4f}  {tm-om:>+8.4f}")

    pm_b = cf_b[:, :, 11] if n_f == 14 else None
    return b_mask, pm_b


def dynamic_topk_calibrate(logits: np.ndarray, cands_np: np.ndarray,
                            labels: np.ndarray, oracle_rank: np.ndarray,
                            oracle_dist: np.ndarray, conf_feats: dict,
                            temp: float = 1.0) -> tuple:
    """Dynamic topk calibration — Rules A-D on seed42 OOF.
    Returns (best_rule_name or None, best_topk_arr or None).
    """
    N = len(labels)
    is_hit  = oracle_dist <= R_HIT_THRESHOLD
    in_t5   = oracle_rank < 5
    ga = is_hit  &  in_t5
    gb = is_hit  & ~in_t5
    gc = ~is_hit

    bt = np.full(N, 10, np.int64)
    bp = _topk_preds_np(logits, cands_np, bt, temp)
    bh = r_hit(bp, labels)
    ba = r_hit(bp[ga], labels[ga])
    bb = r_hit(bp[gb], labels[gb])
    bc = r_hit(bp[gc], labels[gc])
    print(f"  Baseline topk=10: OOF={bh:.4f}  A={ba:.4f}  B={bb:.4f}  C={bc:.4f}")

    gap   = conf_feats["top1_top2_gap"]
    ent5  = conf_feats["top5_entropy"]
    pd_   = conf_feats["phys_dis_cm"]
    q30g  = np.percentile(gap,  30); q70g = np.percentile(gap,  70)
    q30e  = np.percentile(ent5, 30); q40e = np.percentile(ent5, 40)
    q70e  = np.percentile(ent5, 70)
    q70d  = np.percentile(pd_,  70)

    rules = {
        "A(gap)":       np.where(gap  >= q70g,                 3,
                        np.where(gap  <= q30g,                15, 10)).astype(np.int64),
        "B(entropy)":   np.where(ent5 <= q30e,                 3,
                        np.where(ent5 >= q70e,                15, 10)).astype(np.int64),
        "C(gap+ent)":   np.where((gap >= q70g)&(ent5 <= q40e), 3,
                        np.where((gap <= q30g)|(ent5 >= q70e),15, 10)).astype(np.int64),
        "D(C+phys)":    np.where((gap >= q70g)&(ent5 <= q40e), 3,
                        np.where((gap <= q30g)|(ent5 >= q70e)|(pd_ >= q70d),
                                  15, 10)).astype(np.int64),
    }

    print(f"\n  {'rule':<13}  {'OOF':>6}  {'dOOF':>6}  {'A':>6}  {'dA':>6}"
          f"  {'B':>6}  {'dB':>6}  판정")
    print(f"  {'-'*72}")

    best_rule = None; best_ta = None; best_hit = bh

    for rn, ta in rules.items():
        preds = _topk_preds_np(logits, cands_np, ta, temp)
        h = r_hit(preds, labels)
        ha = r_hit(preds[ga], labels[ga]); hb = r_hit(preds[gb], labels[gb])
        n3 = int((ta==3).sum()); n15 = int((ta==15).sum())
        ok = (h >= bh - 0.0001) and (ha >= ba - 0.0005)
        v  = "OK" if (ok and h > best_hit) else ("=" if ok else "NG")
        print(f"  {rn:<13}  {h:.4f}  {h-bh:+.4f}  {ha:.4f}  {ha-ba:+.4f}"
              f"  {hb:.4f}  {hb-bb:+.4f}  {v}")
        print(f"    分布 k=3:{n3}({n3/N:.0%})  k=10:{N-n3-n15}({(N-n3-n15)/N:.0%})"
              f"  k=15:{n15}({n15/N:.0%})")
        if ok and h > best_hit:
            best_hit = h; best_rule = rn; best_ta = ta.copy()

    msg = f"{best_rule}  OOF={best_hit:.4f} ({best_hit-bh:+.4f})" if best_rule else "없음"
    print(f"\n  → Best rule: {msg}")
    return best_rule, best_ta


def soft_reranking_calibrate(logits: np.ndarray, cands_np: np.ndarray,
                              coords: np.ndarray, labels: np.ndarray,
                              oracle_rank: np.ndarray, oracle_dist: np.ndarray,
                              conf_feats: dict, temp: float = 1.0) -> list | None:
    """
    Logit bias sensitivity analysis for B-risk samples.
    Bias applied only where b_risk=True; non-risk samples unchanged.
    Returns list of result dicts (or None if interaction features unavailable).
    """
    from candidates import (make_cand_features as _mcf,  # type: ignore
                             CAND_FEAT_INTERACTION)

    if not CAND_FEAT_INTERACTION:
        print("  ※ CAND_FEAT_INTERACTION=False → interaction feature 없음. 건너뜀.")
        return None

    N = len(labels)
    is_hit  = oracle_dist <= R_HIT_THRESHOLD
    in_t5   = oracle_rank < 5
    ga = is_hit  &  in_t5
    gb = is_hit  & ~in_t5

    # B-risk gate (union of four conditions)
    gap   = conf_feats["top1_top2_gap"]
    ent5  = conf_feats["top5_entropy"]
    perp  = conf_feats["obs_acc_perp_abs"]
    jk_cf = conf_feats["obs_jerk_abs"]
    q40g  = np.percentile(gap,  40); q60e = np.percentile(ent5, 60)
    q70p  = np.percentile(perp, 70); q70j = np.percentile(jk_cf, 70)
    b_risk = (gap <= q40g) | (ent5 >= q60e) | (perp >= q70p) | (jk_cf >= q70j)
    print(f"  B-risk gate: {b_risk.sum()} ({b_risk.mean():.1%}) samples")

    # Precompute interaction features (N, C, 14)
    print("  Interaction feature 계산 중...")
    cf_all     = _mcf(coords, cands_np)              # (N, C, 14)
    par_match  = cf_all[:, :, 11]
    perp_match = cf_all[:, :, 12]
    jerk_match = cf_all[:, :, 13]

    # Case gates based on physics
    p0_, d1_, d2_, acc_, jk_v = motion_terms(coords, end_idx=10)
    sp_  = np.linalg.norm(d1_, axis=1, keepdims=True).clip(EPS)
    tg_  = d1_ / sp_
    aps_ = (acc_ * tg_).sum(axis=1, keepdims=True)
    apv_ = acc_ - aps_ * tg_
    acc_perp_n = np.linalg.norm(apv_, axis=1) / (sp_[:, 0] + EPS)
    jerk_n_    = np.linalg.norm(jk_v,  axis=1) / (sp_[:, 0] + EPS)
    turn_gate  = acc_perp_n >= q70p
    jerk_gate  = jerk_n_    >= q70j

    topk_arr = np.full(N, 10, np.int64)

    def _eval(bias_mat):
        lg = logits.copy()
        lg[b_risk] += bias_mat[b_risk]
        preds = _topk_preds_np(lg, cands_np, topk_arr, temp)
        return (r_hit(preds, labels),
                r_hit(preds[ga], labels[ga]),
                r_hit(preds[gb], labels[gb]))

    bh, ba, bb = _eval(np.zeros((N, N_CANDIDATES), np.float32))
    print(f"  Baseline (no bias): OOF={bh:.4f}  A={ba:.4f}  B={bb:.4f}")

    hdr = (f"  {'bias':<18}  {'w':>4}  {'OOF':>6}  {'dOOF':>6}"
           f"  {'A':>6}  {'dA':>6}  {'B':>6}  {'dB':>6}  판정")
    print(f"\n  Sensitivity (단일 bias):\n{hdr}\n  {'-'*(len(hdr)-2)}")

    bias_defs = [
        ("par_match",  par_match),
        ("perp_match", perp_match),
        ("jerk_match", jerk_match),
        ("turn_bias",  perp_match * turn_gate[:, None]),
        ("jerk_bias",  jerk_match * jerk_gate[:, None]),
    ]
    results = []
    for bname, bbase in bias_defs:
        for w in [0.01, 0.02, 0.04, 0.06]:
            h, ha, hb = _eval(w * bbase)
            dh = h-bh; da = ha-ba; db = hb-bb
            ok = (h >= bh - 0.0001) and (ha >= ba - 0.0005)
            v  = "OK" if (ok and dh > 5e-5) else ("=" if ok else "NG")
            print(f"  {bname:<18}  {w:.2f}  {h:.4f}  {dh:+.4f}"
                  f"  {ha:.4f}  {da:+.4f}  {hb:.4f}  {db:+.4f}  {v}")
            results.append(dict(bias=bname, w=w, hit=h, dh=dh,
                                ha=ha, da=da, hb=hb, db=db, ok=ok, verdict=v))

    # Top-2 combination
    valid = sorted([r for r in results if r["verdict"] == "OK"], key=lambda r: -r["hit"])
    bmap  = {n: b for n, b in bias_defs}
    if len(valid) >= 2 and valid[0]["bias"] != valid[1]["bias"]:
        r1, r2 = valid[0], valid[1]
        bm = r1["w"] * bmap[r1["bias"]] + r2["w"] * bmap[r2["bias"]]
        h_, ha_, hb_ = _eval(bm)
        ok_ = (h_ >= bh - 0.0001) and (ha_ >= ba - 0.0005)
        v_  = "OK" if (ok_ and h_ > bh + 5e-5) else ("=" if ok_ else "NG")
        print(f"\n  Top-2 조합: {r1['bias']}×{r1['w']}+{r2['bias']}×{r2['w']}")
        print(f"  OOF={h_:.4f} ({h_-bh:+.4f})  A={ha_:.4f}  B={hb_:.4f}  {v_}")

    if valid:
        br = valid[0]
        print(f"\n  → Best: {br['bias']} w={br['w']:.2f}  OOF={br['hit']:.4f} ({br['dh']:+.4f})")
    else:
        print("\n  → 유효한 bias 없음. soft reranking 폐기.")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2-B: C-group type analysis + Gate design
# ─────────────────────────────────────────────────────────────────────────────

def c_group_type_analysis(cands_np: np.ndarray, coords: np.ndarray,
                           labels: np.ndarray, oracle_idx: np.ndarray,
                           oracle_dist: np.ndarray) -> dict:
    """
    C-group 샘플을 4개 유형으로 분류:
    1. sharp_turn_C  : nearest = turn family
    2. high_jerk_C   : nearest = jerk family
    3. latency_C     : nearest = latency family
    4. par_out_C     : nearest = base/acc/frenet family
    """
    from candidates import CANDIDATE_FAMILY, FAMILY_NAMES

    N = len(labels)
    c_mask  = oracle_dist > R_HIT_THRESHOLD   # (N,) C-group
    n_c     = int(c_mask.sum())
    fnames  = list(FAMILY_NAMES)
    jerk_fid, turn_fid, lat_fid = fnames.index("jerk"), fnames.index("turn"), fnames.index("latency")

    # Nearest family for ALL samples (oracle_idx = nearest cand even for C)
    near_fam = np.array(CANDIDATE_FAMILY)[oracle_idx]   # (N,)
    near_fam_c = near_fam[c_mask]                        # (N_C,)

    # Physics features for C-group
    p0_, d1_, d2_, acc_, jk_v = motion_terms(coords[c_mask], end_idx=10)
    sp_   = np.linalg.norm(d1_, axis=1, keepdims=True).clip(EPS)
    tg_   = d1_ / sp_
    aps_  = (acc_ * tg_).sum(axis=1, keepdims=True)
    apv_  = acc_ - aps_ * tg_
    obs_jerk_c = np.linalg.norm(jk_v,  axis=1) / (sp_[:, 0] + EPS)
    obs_perp_c = np.linalg.norm(apv_,  axis=1) / (sp_[:, 0] + EPS)
    near_dist_c = (np.linalg.norm(
        cands_np[c_mask] - labels[c_mask, None, :], axis=-1
    ).min(axis=1) * 100)  # cm

    type_masks = {
        "sharp_turn_C": near_fam_c == turn_fid,
        "high_jerk_C":  near_fam_c == jerk_fid,
        "latency_C":    near_fam_c == lat_fid,
        "par_out_C":    ~np.isin(near_fam_c, [turn_fid, jerk_fid, lat_fid]),
    }

    print(f"  C-group: {n_c} ({n_c/N:.1%} of all)")
    print(f"\n  {'C-type':<15}  {'cnt':>5}  {'%_C':>6}  {'%_all':>6}  "
          f"{'jerk_abs':>8}  {'perp_abs':>8}  {'near_cm':>7}")
    print(f"  {'-'*15}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*7}")
    type_info = {}
    for ctype, tm in type_masks.items():
        n = int(tm.sum())
        if n == 0:
            continue
        top_fam = fnames[int(np.bincount(near_fam_c[tm].astype(int)).argmax())]
        print(f"  {ctype:<15}  {n:>5}  {n/n_c:>5.1%}  {n/N:>5.1%}  "
              f"{obs_jerk_c[tm].mean():>8.4f}  "
              f"{obs_perp_c[tm].mean():>8.4f}  "
              f"{near_dist_c[tm].mean():>7.2f}")
        type_info[ctype] = {"n": n, "near_fam": top_fam,
                             "jerk": obs_jerk_c[tm], "perp": obs_perp_c[tm]}

    # Nearest candidate distribution (top-5)
    print(f"\n  C-group nearest 후보 top-5:")
    uniq, cnts = np.unique(oracle_idx[c_mask], return_counts=True)
    from candidates import CANDIDATES
    for idx, cnt in sorted(zip(uniq.tolist(), cnts.tolist()), key=lambda x: -x[1])[:5]:
        print(f"    [{idx:>2}] {CANDIDATES[idx].name:<25}  {cnt:>4} ({cnt/n_c:.1%})")

    return type_info


def gate_design_analysis(oracle_dist: np.ndarray, conf_feats: dict) -> list:
    """
    C-risk gate 후보를 평가한다.
    oracle_dist: (N,) — train/OOF에서만 알 수 있는 값
    conf_feats:  compute_conf_features() 출력
    """
    N = len(oracle_dist)
    c_group = oracle_dist > R_HIT_THRESHOLD   # (N,) bool
    n_c     = int(c_group.sum())

    obs_jerk = conf_feats["obs_jerk_abs"]
    obs_perp = conf_feats["obs_acc_perp_abs"]
    phys_dis = conf_feats["phys_dis_cm"]
    entropy  = conf_feats["top5_entropy"]

    # Print feature distribution by group (for reference)
    ab_group = ~c_group
    print(f"  Feature 분포 (C vs A/B):")
    for fn, fv in [("obs_jerk", obs_jerk), ("obs_perp", obs_perp),
                   ("phys_dis", phys_dis)]:
        c_m = fv[c_group].mean(); ab_m = fv[ab_group].mean()
        c_q90 = np.percentile(fv[c_group], 90)
        all_q90 = np.percentile(fv, 90)
        print(f"    {fn:<10}: C={c_m:.4f}  AB={ab_m:.4f}  "
              f"C_q90={c_q90:.4f}  all_q90={all_q90:.4f}")

    # Gate candidates
    gates: dict[str, np.ndarray] = {}
    for q in [88, 90, 92, 95]:
        gates[f"J_q{q}"]   = obs_jerk >= np.percentile(obs_jerk, q)
        gates[f"T_q{q}"]   = obs_perp >= np.percentile(obs_perp, q)
        gates[f"L_q{q}"]   = phys_dis >= np.percentile(phys_dis, q)
        jq = np.percentile(obs_jerk, q); tq = np.percentile(obs_perp, q)
        gates[f"JT_q{q}"]  = (obs_jerk >= jq) | (obs_perp >= tq)

    # Add entropy filter variants
    q50e = np.percentile(entropy, 50)
    gates["J90_e50"]  = (obs_jerk >= np.percentile(obs_jerk, 90)) & (entropy >= q50e)
    gates["JT90_e50"] = ((obs_jerk >= np.percentile(obs_jerk, 90)) |
                         (obs_perp >= np.percentile(obs_perp, 90))) & (entropy >= q50e)

    print(f"\n  {'gate':<14}  {'act%':>5}  {'C_prec':>7}  {'C_rec':>6}  {'contam':>7}")
    print(f"  {'-'*14}  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*7}")

    viable = []
    for gname, gmask in sorted(gates.items()):
        n_act = int(gmask.sum())
        if n_act == 0:
            continue
        tp   = int((gmask & c_group).sum())
        prec = tp / n_act
        rec  = tp / n_c
        act  = n_act / N
        marker = "  <-- OK" if prec >= 0.45 and act <= 0.10 else ""
        print(f"  {gname:<14}  {act:>4.1%}  {prec:>7.3f}  {rec:>6.3f}  {1-prec:>7.3f}{marker}")
        if prec >= 0.45 and act <= 0.10:
            viable.append(dict(name=gname, mask=gmask, prec=prec, rec=rec, act=act))

    if viable:
        best = max(viable, key=lambda x: x["prec"])
        print(f"\n  Best gate: {best['name']}  "
              f"prec={best['prec']:.3f}  rec={best['rec']:.3f}  act={best['act']:.1%}")
    else:
        print(f"\n  C_precision >= 0.45 만족하는 gate 없음 — threshold 상향 필요")

    return viable


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2-A: Extended B-group jerk over-selection analysis
# ─────────────────────────────────────────────────────────────────────────────

def extended_b_group_jerk_analysis(logits: np.ndarray, cands_np: np.ndarray,
                                    coords: np.ndarray, labels: np.ndarray,
                                    oracle_idx: np.ndarray, oracle_rank: np.ndarray,
                                    oracle_dist: np.ndarray) -> None:
    """
    Deep-dive into B-group jerk over-selection.
    1. Oracle family distribution when top1 = jerk
    2. Jerk / perp sign mismatch analysis
    3. Turn oracle selector rank + logit gap vs jerk top1
    """
    from candidates import (CANDIDATE_FAMILY, FAMILY_NAMES, CANDIDATES,
                             motion_terms)

    is_hit = oracle_dist <= R_HIT_THRESHOLD
    in_t5  = oracle_rank < 5
    b_mask = is_hit & ~in_t5
    n_b    = int(b_mask.sum())

    top1_idx_all  = logits.argmax(axis=1)           # (N,)
    top1_idx_b    = top1_idx_all[b_mask]             # (N_b,)
    oracle_idx_b  = oracle_idx[b_mask]               # (N_b,)
    oracle_rank_b = oracle_rank[b_mask]              # (N_b,)
    logits_b      = logits[b_mask]                   # (N_b, C)
    coords_b      = coords[b_mask]
    labels_b      = labels[b_mask]

    fam_arr    = np.array(CANDIDATE_FAMILY)
    fam_names  = list(FAMILY_NAMES)
    jerk_fid   = fam_names.index("jerk")
    turn_fid   = fam_names.index("turn")

    top1_fam   = fam_arr[top1_idx_b]
    oracle_fam = fam_arr[oracle_idx_b]
    jerk_top1  = top1_fam == jerk_fid
    n_jt1      = int(jerk_top1.sum())

    # ── 1. Oracle family distribution when top1=jerk ─────────────────────────
    print(f"\n  [1] Oracle family when top1=jerk ({n_jt1}/{n_b} = {n_jt1/n_b:.1%})")
    print(f"  {'family':<10}  {'count':>6}  {'ratio':>6}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*6}")
    for fid, fname in enumerate(fam_names):
        cnt = int((oracle_fam[jerk_top1] == fid).sum())
        print(f"  {fname:<10}  {cnt:>6}  {cnt/n_jt1:>5.1%}")

    # ── 2. Sign mismatch analysis ─────────────────────────────────────────────
    top1_jerk_p   = np.array([CANDIDATES[i].jerk for i in top1_idx_b])   # (N_b,)
    oracle_jerk_p = np.array([CANDIDATES[i].jerk for i in oracle_idx_b]) # (N_b,)
    top1_perp_p   = np.array([CANDIDATES[i].perp for i in top1_idx_b])
    oracle_perp_p = np.array([CANDIDATES[i].perp for i in oracle_idx_b])

    p0_, d1_, d2_, acc_, jk_v = motion_terms(coords_b, end_idx=10)
    sp_     = np.linalg.norm(d1_, axis=1, keepdims=True).clip(EPS)
    tg_     = d1_ / sp_
    obs_jk_par = (jk_v * tg_).sum(axis=1) / (sp_[:, 0] + EPS)  # (N_b,) signed

    jk_sm_t1 = np.sign(top1_jerk_p)   * np.sign(obs_jk_par)  # -1/0/+1
    jk_sm_or = np.sign(oracle_jerk_p)  * np.sign(obs_jk_par)

    print(f"\n  [2] Jerk sign match (B-group N={n_b}):")
    print(f"  {'sign_match':>12}  {'top1':>8}  {'oracle':>8}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*8}")
    for label, sm in [("match (+1)", 1), ("mismatch(-1)", -1), ("zero (0)", 0)]:
        n_t1 = int((jk_sm_t1 == sm).sum()); n_or = int((jk_sm_or == sm).sum())
        print(f"  {label:>12}  {n_t1:>7} ({n_t1/n_b:.1%})  {n_or:>7} ({n_or/n_b:.1%})")

    # turn oracle + jerk top1 subset
    turn_or_jerk_t1 = jerk_top1 & (oracle_fam == turn_fid)
    n_tojt = int(turn_or_jerk_t1.sum())
    print(f"\n  turn oracle + jerk top1: {n_tojt} ({n_tojt/n_b:.1%})")

    if n_tojt > 0:
        jk_mismatch_sub = (np.sign(top1_jerk_p[turn_or_jerk_t1]) *
                           np.sign(obs_jk_par[turn_or_jerk_t1])) < 0
        or_perp_sub = oracle_perp_p[turn_or_jerk_t1]
        print(f"    jerk sign mismatch rate  : {jk_mismatch_sub.mean():.1%}")
        print(f"    oracle perp > 0 (turn_p*): {(or_perp_sub > 0).mean():.1%}")
        print(f"    oracle perp < 0 (turn_n*): {(or_perp_sub < 0).mean():.1%}")

    # perp sign: how often does top1 perp sign match oracle perp sign?
    # (relevant when both have nonzero perp)
    nonzero_perp = (top1_perp_p != 0) & (oracle_perp_p != 0)
    if nonzero_perp.sum() > 0:
        perp_sign_agree = (np.sign(top1_perp_p[nonzero_perp]) ==
                           np.sign(oracle_perp_p[nonzero_perp]))
        print(f"\n  Perp sign agreement when both nonzero ({nonzero_perp.sum()} samples):")
        print(f"    agree   : {perp_sign_agree.mean():.1%}")
        print(f"    disagree: {(~perp_sign_agree).mean():.1%}")

    # ── 3. Turn oracle rank + logit gap vs jerk top1 ─────────────────────────
    turn_or_mask = oracle_fam == turn_fid   # B-group with turn oracle
    n_to = int(turn_or_mask.sum())
    rank_to = oracle_rank_b[turn_or_mask]

    print(f"\n  [3] Turn oracle rank (B-group, N={n_to}):")
    for k in [1, 3, 5, 10, 15, 20, 30]:
        nk = int((rank_to < k).sum())
        print(f"    rank < {k:>2}: {nk:4d} ({nk/n_to:.1%})")
    print(f"    mean={rank_to.mean():.1f}  median={np.median(rank_to):.0f}"
          f"  p75={np.percentile(rank_to,75):.0f}  p90={np.percentile(rank_to,90):.0f}")

    # Logit gap: jerk top1 logit - turn oracle logit (subset: jerk top1 & turn oracle)
    if n_tojt > 0:
        idx_sub = np.where(turn_or_jerk_t1)[0]
        top1_lgts = logits_b[idx_sub, top1_idx_b[turn_or_jerk_t1]]
        or_lgts   = logits_b[idx_sub, oracle_idx_b[turn_or_jerk_t1]]
        gap_sub   = top1_lgts - or_lgts
        print(f"\n  Logit gap: jerk(top1) - turn(oracle) [N={n_tojt}]:")
        print(f"    mean={gap_sub.mean():.4f}  std={gap_sub.std():.4f}"
              f"  p25={np.percentile(gap_sub,25):.3f}"
              f"  p50={np.percentile(gap_sub,50):.3f}"
              f"  p75={np.percentile(gap_sub,75):.3f}")
        print(f"    top1 wins by >{gap_sub.max():.2f}  median win={np.median(gap_sub):.2f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  Summary:")
    print(f"  B-group: {n_b}  ({n_b/len(labels):.1%})")
    print(f"  jerk top1 rate      : {n_jt1/n_b:.1%}  (oracle jerk: "
          f"{(oracle_fam==jerk_fid).mean():.1%})")
    print(f"  turn oracle rate    : {(oracle_fam==turn_fid).mean():.1%}  (top1 turn: "
          f"{(top1_fam==turn_fid).mean():.1%})")
    print(f"  turn_or+jerk_t1 case: {n_tojt/n_b:.1%}  → core mis-ranking target")


def analyze(seeds: list = None, out_tag: str = ""):
    if seeds is None:
        seeds = [SEED]
    seed = seeds[0]  # primary seed for sections 1-6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids, coords, labels = load_all(TRAIN_DIR, LABELS_PATH)

    # ── 실험 설정 요약 ────────────────────────────────────────────
    print("=" * 55)
    print("EXPERIMENT CONFIG")
    print("=" * 55)
    seeds_str = str(seeds) if len(seeds) > 1 else str(seed)
    print(f"  seeds={seeds_str}  |  N_FOLDS={N_FOLDS}  |  N_CANDIDATES={N_CANDIDATES}")
    print(f"  AUG_MODE={AUG_MODE}", end="")
    if AUG_MODE == 'yaw_speed':
        print(f"  scale={SPEED_SCALE_RANGE}  prob={SPEED_SCALE_PROB}", end="")
    print()
    print(f"  TOPK={TOPK}  |  LISTMLE_WEIGHT={LISTMLE_WEIGHT}  |  PAIRWISE_WEIGHT={PAIRWISE_WEIGHT}")
    print(f"  D_MODEL={D_MODEL}  |  NUM_LAYERS={NUM_LAYERS}  |  DROPOUT={DROPOUT}")
    if out_tag:
        print(f"  out_tag={out_tag!r}")
    print()

    # Load primary seed models for sections 1-6
    model_dir = OUTPUT_DIR / f"seed{seed}{out_tag}"
    print(f"모델 디렉토리: {model_dir}  (seed={seed})")

    models = {}
    for fold in range(N_FOLDS):
        path = model_dir / f"selector_fold{fold}.pt"
        if path.exists():
            m = CandidateSelector().to(device)
            m.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=False)
            m.eval()
            models[fold] = m
    if not models:
        print(f"저장된 모델 없음. python train.py --seed {seed} 먼저 실행.")
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

    # ── 9-12. Stage 1: B-group + Confidence + Calibration ───────
    print("\n" + "=" * 55)
    print("9. B-GROUP DIAGNOSTIC  (seed=%s)" % seed)
    print("=" * 55)
    oof_logits, cands_np_s1 = get_oof_logits_cands(models, ids, coords, device)
    dist_c_s1  = np.linalg.norm(cands_np_s1 - labels[:, None, :], axis=-1)
    oi_s1      = dist_c_s1.min(axis=1)        # oracle dist  (N,)
    oidx_s1    = dist_c_s1.argmin(axis=1)     # oracle cand idx (N,)
    sli_s1     = np.argsort(-oof_logits, axis=1)
    orank_s1   = (sli_s1 == oidx_s1[:, None]).argmax(axis=1)  # oracle rank (N,)
    b_mask_s1, _ = analyze_b_group(
        oof_logits, cands_np_s1, coords, labels, oidx_s1, orank_s1, oi_s1
    )

    print("\n" + "=" * 55)
    print("10. CONFIDENCE FEATURES  (A/B/C 비교, seed=%s)" % seed)
    print("=" * 55)
    conf_s1 = compute_conf_features(oof_logits, cands_np_s1, coords)
    is_hit_s1  = oi_s1  <= R_HIT_THRESHOLD
    in_t5_s1   = orank_s1 < 5
    ga_s1 = is_hit_s1  &  in_t5_s1
    gb_s1 = is_hit_s1  & ~in_t5_s1
    gc_s1 = ~is_hit_s1
    print(f"  {'feature':<20}  {'A':>8}  {'B':>8}  {'C':>8}  {'B-A':>7}  p25~p75(B)")
    print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*15}")
    for fn, fv in conf_s1.items():
        am = fv[ga_s1].mean(); bm = fv[gb_s1].mean(); cm = fv[gc_s1].mean()
        p25b = np.percentile(fv[gb_s1], 25); p75b = np.percentile(fv[gb_s1], 75)
        print(f"  {fn:<20}  {am:>8.4f}  {bm:>8.4f}  {cm:>8.4f}  {bm-am:>+7.4f}"
              f"  [{p25b:.3f}, {p75b:.3f}]")

    print("\n" + "=" * 55)
    print("11. DYNAMIC TOP-K CALIBRATION  (seed=%s OOF)" % seed)
    print("=" * 55)
    best_rule_s1, best_ta_s1 = dynamic_topk_calibrate(
        oof_logits, cands_np_s1, labels, orank_s1, oi_s1, conf_s1
    )

    print("\n" + "=" * 55)
    print("12. SOFT RERANKING BIAS  (B-risk, sensitivity, seed=%s)" % seed)
    print("=" * 55)
    bias_results_s1 = soft_reranking_calibrate(
        oof_logits, cands_np_s1, coords, labels, orank_s1, oi_s1, conf_s1
    )

    print("\n" + "=" * 55)
    print("13. EXTENDED B-GROUP JERK ANALYSIS  (Stage 2-A prep)")
    print("=" * 55)
    extended_b_group_jerk_analysis(
        oof_logits, cands_np_s1, coords, labels, oidx_s1, orank_s1, oi_s1
    )

    # ── 14. C-group type analysis ──────────────────────────────────
    print("\n" + "=" * 55)
    print("14. C-GROUP TYPE ANALYSIS  (Stage 2-B prep)")
    print("=" * 55)
    c_group_type_analysis(cands_np_s1, coords, labels, oidx_s1, oi_s1)

    # ── 15. Gate design ────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("15. C-RISK GATE DESIGN  (Stage 2-B)")
    print("=" * 55)
    gate_design_analysis(oi_s1, conf_s1)

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

    # ── 7. Multi-seed ensemble OOF ────────────────────────────────
    if len(seeds) > 1:
        n_total = len(seeds) * N_FOLDS
        print("\n" + "=" * 55)
        print(f"7. MULTI-SEED ENSEMBLE OOF  ({len(seeds)} seeds × {N_FOLDS} folds = {n_total} models)")
        print("=" * 55)
        print("  모델 로드 중...")
        all_models = load_all_models(seeds, device)
        n_loaded = sum(len(v) for v in all_models.values())
        if n_loaded < 2:
            print("  로드된 모델 부족. 각 seed 학습 완료 후 재실행.")
        else:
            print(f"  총 {n_loaded}개 모델 로드 완료\n")
            ms_preds = get_oof_preds_multiseed(
                all_models, ids, coords, device, topk=best_k, temp=best_temp
            )
            ms_hit   = r_hit(ms_preds, labels)
            print(f"  Single-seed OOF (seed={seed})    : {best_hit:.4f}")
            print(f"  Multi-seed  OOF ({len(seeds)} seeds)  : {ms_hit:.4f}  ({(ms_hit - best_hit)*100:+.2f}pp)")
            print(f"  Oracle ceiling                 : {oracle:.4f}")
            print(f"  Multi-seed efficiency          : {ms_hit/oracle:.1%} of oracle")

    # ── 8. Regression blend (RegMLP OOF가 있을 때) ───────────────
    reg_oof_paths = [OUTPUT_DIR / f"seed{s}" / "regmlp_oof.npy" for s in seeds]
    available = [p for p in reg_oof_paths if p.exists()]
    if available:
        print("\n" + "=" * 55)
        print(f"8. REGRESSION ENTROPY BLEND  (β grid search)")
        print("=" * 55)

        # 회귀 OOF: 여러 seed 평균
        reg_preds_list = [np.load(p) for p in available]
        reg_oof = np.mean(reg_preds_list, axis=0)               # (N, 3)
        reg_hit = r_hit(reg_oof, labels)
        print(f"  Regression OOF (단독)       : {reg_hit:.4f}")

        # Selector OOF logits (entropy 계산용)
        print("  Selector OOF logits 계산 중...")
        sel_logits = get_oof_logits(models, ids, coords, device)  # (N, C)
        sel_preds  = topk_preds[best_k]                            # (N, 3)
        sel_hit    = r_hit(sel_preds, labels)

        # Entropy 분포 확인
        probs_np = np.exp(sel_logits - sel_logits.max(axis=1, keepdims=True))
        probs_np /= probs_np.sum(axis=1, keepdims=True)
        H = -np.sum(probs_np * np.log(probs_np + 1e-9), axis=1) / np.log(N_CANDIDATES)
        print(f"  Entropy H: mean={H.mean():.3f}  p25={np.percentile(H,25):.3f}"
              f"  p75={np.percentile(H,75):.3f}  p95={np.percentile(H,95):.3f}")

        # β grid search
        print(f"\n  {'β':>5}  {'Blend OOF':>10}  {'vs Selector':>12}")
        print(f"  {'-'*5}  {'-'*10}  {'-'*12}")
        best_beta, best_blend_hit = 0.0, sel_hit
        for beta in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]:
            blended = entropy_blend(sel_preds, reg_oof, sel_logits, beta=beta)
            bh = r_hit(blended, labels)
            marker = "  ← best" if bh > best_blend_hit else ""
            if bh > best_blend_hit:
                best_blend_hit, best_beta = bh, beta
            print(f"  {beta:>5.1f}  {bh:.4f}      {(bh-sel_hit)*100:+.2f}pp{marker}")

        print(f"\n  Selector OOF            : {sel_hit:.4f}")
        print(f"  Best blend (β={best_beta:.1f}) OOF : {best_blend_hit:.4f}"
              f"  ({(best_blend_hit-sel_hit)*100:+.2f}pp)")
        print(f"  → 제출 권장: python predict.py --seeds {' '.join(map(str,seeds))} --beta {best_beta:.1f}")
    else:
        print("\n  [섹션 8 skip] RegMLP OOF 없음."
              f" 먼저 실행: python regression.py --seed {seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--seed",  type=int, default=None,
                       help="Single seed to analyze (default: config.SEED)")
    group.add_argument("--seeds", type=int, nargs="+", default=None,
                       help="Multiple seeds for ensemble OOF, e.g. --seeds 42 123 777")
    parser.add_argument("--out-tag", type=str, default="",
                        help="Suffix appended to model dir (e.g. '_lml05'). Must match --out-tag used in train.py.")
    parser.add_argument("--sign-feat", action="store_true",
                        help="Enable sign features (CAND_DIM 14->16). Use with _stage2a_sign/combo models.")
    args = parser.parse_args()

    # Stage 2-A: patch CAND_DIM if sign features were used during training
    if args.sign_feat:
        import config as _cfg
        import candidates as _cands
        import model as _mdl
        _cfg.CAND_FEAT_SIGN   = True
        _cands.CAND_FEAT_SIGN = True
        # CandidateSelector is already imported at top; patch CAND_DIM used in __init__
        _mdl.CAND_DIM = 16

    if args.seeds:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [SEED]

    analyze(seeds=seeds, out_tag=args.out_tag)
