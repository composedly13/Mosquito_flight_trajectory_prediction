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


def load_all_models(seeds: list, device: torch.device) -> dict:
    """Load fold models for multiple seeds. Returns {seed: {fold_idx: model}}."""
    all_models = {}
    for seed in seeds:
        model_dir = OUTPUT_DIR / f"seed{seed}"
        fold_models = {}
        for fold in range(N_FOLDS):
            path = model_dir / f"selector_fold{fold}.pt"
            if path.exists():
                m = CandidateSelector().to(device)
                m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                m.eval()
                fold_models[fold] = m
        if fold_models:
            all_models[seed] = fold_models
            print(f"  seed{seed}: {len(fold_models)}/{N_FOLDS} folds 로드")
        else:
            print(f"  seed{seed}: 모델 없음 (python train.py --seed {seed} 먼저 실행)")
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


def analyze(seeds: list = None):
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
    print()

    # Load primary seed models for sections 1-6
    model_dir = OUTPUT_DIR / f"seed{seed}"
    print(f"모델 디렉토리: {model_dir}  (seed={seed})")

    models = {}
    for fold in range(N_FOLDS):
        path = model_dir / f"selector_fold{fold}.pt"
        if path.exists():
            m = CandidateSelector().to(device)
            m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
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
    args = parser.parse_args()

    if args.seeds:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [SEED]

    analyze(seeds=seeds)
