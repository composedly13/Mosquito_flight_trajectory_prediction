"""
Stage 1: Dynamic Top-K Inference  (재학습 불필요)

샘플별 confidence(entropy, logit gap, candidate spread)에 따라 topk를 동적으로 선택.

Usage:
  python predict_dynamic.py --calibrate --seeds 42 777
      → OOF 기반 임계값 탐색 (seed42+777 앙상블 logit 기준)

  python predict_dynamic.py --seeds 42 777
      → 기본 설정 (t_low=0.90, t_high=0.96, k_low=5, k_high=15) 제출 파일 생성

  python predict_dynamic.py --seeds 42 777 \\
      --t-low 0.90 --t-high 0.96 --k-low 5 --k-high 15 \\
      --out-name submission_dyntopk.csv
      → 수동 설정으로 제출 파일 생성

Confidence 지표:
  H_norm  : 정규화된 entropy  (낮을수록 confident)
  gap     : top1 - top2 logit gap (높을수록 confident)
  combined: 0.7 × (1 - H_norm) + 0.3 × sigmoid_gap

topk 결정:
  combined > t_high → k_low   (high conf: topk=5, 불필요 후보 제거)
  combined < t_low  → k_high  (low conf:  topk=15, 넓게 탐색)
  else              → k_normal (topk=10, 기본값 유지)

출력:
  outputs/submission_dyntopk_<tag>.csv
  outputs/grid/calibration_results.csv  (--calibrate 시)
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.special import expit  # sigmoid

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    TEST_DIR, TRAIN_DIR, LABELS_PATH, SUBMISSION_PATH,
    OUTPUT_DIR, SEED, N_FOLDS, BATCH_SIZE, TOPK,
)
from dataset import load_all
from model import CandidateSelector
from candidates import (
    make_candidates_gpu, make_seq_features_gpu, make_cand_features_gpu,
    N_CANDIDATES,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants / defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TEMP     = 2.0
DEFAULT_T_LOW    = 0.90   # combined confidence: low → topk = k_high
DEFAULT_T_HIGH   = 0.96   # combined confidence: high → topk = k_low
DEFAULT_K_LOW    = 5      # topk for high-confidence samples
DEFAULT_K_NORMAL = 10     # topk for normal samples
DEFAULT_K_HIGH   = 15     # topk for low-confidence samples


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def fold_id(sample_id: str, n_folds: int = N_FOLDS) -> int:
    return int(hashlib.md5(sample_id.encode()).hexdigest()[:8], 16) % n_folds


def r_hit(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(pred - true, axis=-1) <= 0.01))


# ─────────────────────────────────────────────────────────────────────────────
# Confidence computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_confidence(
    logits: np.ndarray,
    cands:  np.ndarray,
    temp:   float = DEFAULT_TEMP,
) -> dict[str, np.ndarray]:
    """
    logits : (N, C)  — raw selector logits
    cands  : (N, C, 3) — candidate positions
    Returns dict with per-sample confidence arrays (all in [0, 1] or normalized).

    Keys:
      H_norm   : normalized entropy (0=concentrated, 1=uniform)
      gap      : (top1 - top2) logit gap, positive
      spread   : weighted std of top-10 candidate positions (normalized by speed×2)
      combined : 0.7×(1−H_norm) + 0.3×sigmoid(gap/4) ∈ [0,1]
    """
    N, C = logits.shape

    # ── Softmax probabilities (with prediction temp) ─────────────────────────
    logits_shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits_shifted * (1.0 / temp))
    probs /= probs.sum(axis=1, keepdims=True)

    # ── H_norm: normalized entropy ───────────────────────────────────────────
    H = -np.sum(probs * np.log(probs + 1e-9), axis=1) / np.log(C)  # (N,)

    # ── gap: top1 - top2 raw logit gap ───────────────────────────────────────
    sorted_logits = np.sort(logits, axis=1)[:, ::-1]  # descending
    gap = sorted_logits[:, 0] - sorted_logits[:, 1]    # (N,)

    # ── spread: weighted std of top-10 candidate positions ───────────────────
    k_spread = min(10, C)
    top_idx   = np.argsort(-probs, axis=1)[:, :k_spread]    # (N, k_spread)
    top_probs = probs[np.arange(N)[:, None], top_idx]       # (N, k_spread)
    top_probs_n = top_probs / (top_probs.sum(axis=1, keepdims=True) + 1e-8)
    top_cands = cands[np.arange(N)[:, None], top_idx]       # (N, k_spread, 3)

    # Weighted mean candidate
    w_mean = (top_cands * top_probs_n[:, :, None]).sum(axis=1)  # (N, 3)
    diff   = top_cands - w_mean[:, None, :]                      # (N, k, 3)
    spread_m = np.sqrt(
        (top_probs_n[:, :, None] * diff ** 2).sum(axis=(1, 2)) + 1e-8
    )  # (N,) in meters

    # Normalize spread by typical displacement scale (assume ~0.1m = 2×speed)
    # Use norm of mean displacement as proxy for speed×2
    disp_scale = np.linalg.norm(w_mean, axis=1).clip(0.001)  # (N,)
    spread_norm = (spread_m / disp_scale).clip(0, 1)          # (N,) in [0,1]

    # ── Combined confidence score ─────────────────────────────────────────────
    # high value = high confidence
    gap_sig    = expit(gap / 4.0)               # sigmoid(gap/4) → (0,1)
    conf_h     = 1.0 - H                        # entropy inverted
    combined   = 0.7 * conf_h + 0.3 * gap_sig  # (N,)

    return {
        "H_norm":   H,
        "gap":      gap,
        "spread":   spread_norm,
        "combined": combined,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic topk prediction (vectorized, GPU)
# ─────────────────────────────────────────────────────────────────────────────

def selector_predict_dynamic(
    logits:        torch.Tensor,    # (B, C)
    cands:         torch.Tensor,    # (B, C, 3)
    topk_arr:      torch.Tensor,    # (B,) int — per-sample topk
    temp:          float = DEFAULT_TEMP,
) -> torch.Tensor:                  # (B, 3)
    """Vectorized dynamic top-k weighted average."""
    B, C, _ = cands.shape
    weights   = F.softmax(logits / temp, dim=-1)           # (B, C)
    sorted_w, sorted_idx = weights.sort(dim=-1, descending=True)  # (B, C)
    sorted_cands = cands.gather(
        1, sorted_idx.unsqueeze(-1).expand(-1, -1, 3)
    )  # (B, C, 3)

    # Mask: keep only the first topk_arr[i] entries per sample
    arange = torch.arange(C, device=logits.device).unsqueeze(0)   # (1, C)
    mask   = arange < topk_arr.unsqueeze(1)                        # (B, C)

    masked_w = sorted_w * mask.float()
    masked_w = masked_w / (masked_w.sum(dim=-1, keepdim=True) + 1e-8)

    return (sorted_cands * masked_w.unsqueeze(-1)).sum(dim=1)      # (B, 3)


def assign_topk(
    conf:     np.ndarray,    # combined confidence (N,)
    t_low:    float = DEFAULT_T_LOW,
    t_high:   float = DEFAULT_T_HIGH,
    k_low:    int   = DEFAULT_K_LOW,
    k_normal: int   = DEFAULT_K_NORMAL,
    k_high:   int   = DEFAULT_K_HIGH,
) -> np.ndarray:
    """Per-sample topk assignment based on confidence score."""
    topk = np.full(len(conf), k_normal, dtype=np.int64)
    topk[conf >= t_high] = k_low   # high confidence → smaller topk
    topk[conf <  t_low]  = k_high  # low confidence  → larger topk
    return topk


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_selectors(seeds: list[int], device: torch.device) -> dict[int, list]:
    """Returns {seed: [fold0_model, ..., fold4_model]}"""
    seed_models: dict[int, list] = {}
    for seed in seeds:
        seed_dir = OUTPUT_DIR / f"seed{seed}"
        models   = []
        for fold in range(N_FOLDS):
            path = seed_dir / f"selector_fold{fold}.pt"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing: {path}\n  Run: python train.py --seed {seed}"
                )
            m = CandidateSelector().to(device)
            m.load_state_dict(torch.load(path, map_location=device), strict=False)
            m.eval()
            models.append(m)
        seed_models[seed] = models
        print(f"  seed{seed}: {len(models)} fold models loaded")
    return seed_models


# ─────────────────────────────────────────────────────────────────────────────
# OOF logit computation
# ─────────────────────────────────────────────────────────────────────────────

def get_oof_logits_multiseed(
    seed_models: dict[int, list],
    ids:         list[str],
    coords:      np.ndarray,
    device:      torch.device,
    seed_weights: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute OOF logits (ensemble over seeds, per-fold OOF structure).
    Returns: (oof_logits (N,C), oof_cands (N,C,3))
    """
    seeds = list(seed_models.keys())
    if seed_weights is None:
        seed_weights = [1.0 / len(seeds)] * len(seeds)
    seed_weights = np.array(seed_weights) / sum(seed_weights)

    N = len(ids)
    fold_ids = np.array([fold_id(i) for i in ids])
    oof_logits = np.zeros((N, N_CANDIDATES), dtype=np.float32)
    oof_cands  = np.zeros((N, N_CANDIDATES, 3), dtype=np.float32)

    for fi in range(N_FOLDS):
        val_mask = (fold_ids == fi)
        if not val_mask.any():
            continue
        val_c = coords[val_mask]
        val_i = np.where(val_mask)[0]

        fold_logits_sum = np.zeros((val_mask.sum(), N_CANDIDATES), dtype=np.float32)
        fold_cands_set  = None

        for s_idx, (seed, sw) in enumerate(zip(seeds, seed_weights)):
            model = seed_models[seed][fi]
            model.eval()
            seed_logits_list = []
            seed_cands_list  = []

            for start in range(0, len(val_c), BATCH_SIZE):
                end  = min(start + BATCH_SIZE, len(val_c))
                c    = torch.tensor(val_c[start:end]).to(device)
                with torch.no_grad():
                    cands_t = make_candidates_gpu(c)
                    seq_t   = make_seq_features_gpu(c)
                    cand_t  = make_cand_features_gpu(c, cands_t)
                    logits  = model(seq_t, cand_t)
                seed_logits_list.append(logits.cpu().numpy())
                seed_cands_list.append(cands_t.cpu().numpy())

            seed_logits_np = np.concatenate(seed_logits_list, axis=0)  # (nval, C)
            fold_logits_sum += sw * seed_logits_np

            if fold_cands_set is None:
                fold_cands_set = np.concatenate(seed_cands_list, axis=0)  # (nval, C, 3)

        oof_logits[val_i] = fold_logits_sum
        oof_cands[val_i]  = fold_cands_set

    return oof_logits, oof_cands


# ─────────────────────────────────────────────────────────────────────────────
# OOF Calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate(
    seed_models: dict[int, list],
    ids:         list[str],
    coords:      np.ndarray,
    labels:      np.ndarray,
    device:      torch.device,
    seed_weights: list[float] | None = None,
    temp:         float = DEFAULT_TEMP,
) -> dict:
    """
    Grid search over (t_low, t_high, k_low, k_high) on OOF logits.
    Returns best config dict.
    """
    print("\n[OOF logit 계산 중...]")
    oof_logits, oof_cands = get_oof_logits_multiseed(
        seed_models, ids, coords, device, seed_weights
    )

    print("[Confidence 계산 중...]")
    conf_dict = compute_confidence(oof_logits, oof_cands, temp=temp)
    combined  = conf_dict["combined"]

    # Baseline: static topk=10
    N = len(ids)
    def static_pred(topk: int) -> np.ndarray:
        cands_t = torch.tensor(oof_cands).cpu()
        logits_t = torch.tensor(oof_logits).cpu()
        topk_arr = torch.full((N,), topk, dtype=torch.long)
        return selector_predict_dynamic(logits_t, cands_t, topk_arr, temp).numpy()

    baseline_hit = r_hit(static_pred(TOPK), labels)
    print(f"\n  Baseline (topk={TOPK}, temp={temp}): {baseline_hit:.4f}")

    # Group analysis (A/B/C)
    from candidates import make_candidates
    cands_np  = make_candidates(coords)
    dist_c    = np.linalg.norm(cands_np - labels[:, None, :], axis=-1)
    oracle_dist = dist_c.min(axis=1)
    oracle_idx  = dist_c.argmin(axis=1)
    is_oracle_hit = oracle_dist <= 0.01

    sorted_logit_idx = np.argsort(-oof_logits, axis=1)
    oracle_rank = (sorted_logit_idx == oracle_idx[:, None]).argmax(axis=1)
    oracle_in_top5 = oracle_rank < 5

    group_a = is_oracle_hit  &  oracle_in_top5    # oracle hit AND top-5
    group_b = is_oracle_hit  & ~oracle_in_top5    # oracle hit BUT not top-5
    group_c = ~is_oracle_hit                      # no oracle

    print(f"\n  A(oracle & top-5): {group_a.sum()} ({group_a.mean():.1%})")
    print(f"  B(oracle, not top-5): {group_b.sum()} ({group_b.mean():.1%})")
    print(f"  C(no oracle):       {group_c.sum()} ({group_c.mean():.1%})")

    # Confidence distribution
    for label_g, mask_g in [("A", group_a), ("B", group_b), ("C", group_c)]:
        h_g = combined[mask_g]
        print(f"  Conf [{label_g}]: mean={h_g.mean():.3f}  "
              f"p25={np.percentile(h_g,25):.3f}  "
              f"p50={np.percentile(h_g,50):.3f}  "
              f"p75={np.percentile(h_g,75):.3f}")

    # Grid search
    t_lows   = [0.83, 0.86, 0.88, 0.90, 0.92]
    t_highs  = [0.94, 0.96, 0.97, 0.98]
    k_lows   = [3, 5]
    k_highs  = [12, 15, 20]

    print(f"\n[Grid search: {len(t_lows)*len(t_highs)*len(k_lows)*len(k_highs)} 조합]")

    cands_t  = torch.tensor(oof_cands).cpu()
    logits_t = torch.tensor(oof_logits).cpu()

    results = []
    for t_l in t_lows:
        for t_h in t_highs:
            if t_h <= t_l:
                continue
            for k_l in k_lows:
                for k_h in k_highs:
                    topk_np  = assign_topk(combined, t_l, t_h, k_l, DEFAULT_K_NORMAL, k_h)
                    topk_t   = torch.tensor(topk_np, dtype=torch.long)
                    preds    = selector_predict_dynamic(logits_t, cands_t, topk_t, temp).numpy()
                    hit      = r_hit(preds, labels)

                    n_low    = int((topk_np == k_h).sum())
                    n_high   = int((topk_np == k_l).sum())
                    n_normal = N - n_low - n_high

                    results.append({
                        "t_low":   t_l,
                        "t_high":  t_h,
                        "k_low":   k_l,
                        "k_high":  k_h,
                        "oof_hit": hit,
                        "delta":   hit - baseline_hit,
                        "n_low_conf":  n_low,
                        "n_high_conf": n_high,
                        "n_normal":    n_normal,
                        "pct_low_conf":  n_low / N,
                        "pct_high_conf": n_high / N,
                    })

    results.sort(key=lambda r: -r["oof_hit"])

    print(f"\n{'t_low':>6}  {'t_high':>6}  {'k_low':>5}  {'k_high':>6}  "
          f"{'OOF':>7}  {'Δ':>6}  {'%lowC':>6}  {'%highC':>7}")
    print("-" * 75)
    for r in results[:15]:
        print(f"  {r['t_low']:.2f}    {r['t_high']:.2f}    "
              f"{r['k_low']:>3}     {r['k_high']:>4}    "
              f"{r['oof_hit']:.4f}  "
              f"{r['delta']:+.4f}  "
              f"{r['pct_low_conf']:.1%}    {r['pct_high_conf']:.1%}")

    best = results[0]
    print(f"\n  → Best: t_low={best['t_low']:.2f}, t_high={best['t_high']:.2f}, "
          f"k_low={best['k_low']}, k_high={best['k_high']}")
    print(f"  → OOF:  {best['oof_hit']:.4f}  ({best['delta']:+.4f} vs static topk={TOPK})")

    # Per-group analysis with best config
    best_topk_np = assign_topk(combined,
        best["t_low"], best["t_high"], best["k_low"], DEFAULT_K_NORMAL, best["k_high"])
    best_topk_t  = torch.tensor(best_topk_np, dtype=torch.long)
    best_preds   = selector_predict_dynamic(logits_t, cands_t, best_topk_t, temp).numpy()

    print("\n  Per-group R-Hit (best dynamic vs static topk=10):")
    for g_label, g_mask in [("A", group_a), ("B", group_b), ("C", group_c)]:
        if not g_mask.any():
            continue
        s_preds = static_pred(TOPK)
        h_stat = r_hit(s_preds[g_mask], labels[g_mask])
        h_dyn  = r_hit(best_preds[g_mask], labels[g_mask])
        print(f"    {g_label}: static={h_stat:.4f}  dynamic={h_dyn:.4f}  "
              f"Δ={h_dyn-h_stat:+.4f}")

    # Save calibration results
    cal_dir = OUTPUT_DIR / "grid"
    cal_dir.mkdir(parents=True, exist_ok=True)
    cal_path = cal_dir / "calibration_results.csv"
    pd.DataFrame(results).to_csv(cal_path, index=False)
    print(f"\n  Calibration CSV saved: {cal_path}")

    return best


# ─────────────────────────────────────────────────────────────────────────────
# Test inference (submission generation)
# ─────────────────────────────────────────────────────────────────────────────

def predict_test(
    seed_models:  dict[int, list],
    ids:          list[str],
    coords:       np.ndarray,
    device:       torch.device,
    seed_weights: list[float] | None = None,
    t_low:        float = DEFAULT_T_LOW,
    t_high:       float = DEFAULT_T_HIGH,
    k_low:        int   = DEFAULT_K_LOW,
    k_normal:     int   = DEFAULT_K_NORMAL,
    k_high:       int   = DEFAULT_K_HIGH,
    temp:         float = DEFAULT_TEMP,
    out_name:     str   = "",
) -> None:
    """Generate test submission with dynamic topk."""
    seeds        = list(seed_models.keys())
    if seed_weights is None:
        seed_weights = [1.0 / len(seeds)] * len(seeds)
    seed_weights = np.array(seed_weights) / sum(seed_weights)

    N = len(ids)
    print(f"\n[Test inference]  N={N}  seeds={seeds}  "
          f"temp={temp}  t_low={t_low}  t_high={t_high}  "
          f"k_low={k_low}  k_normal={k_normal}  k_high={k_high}")

    all_logits = np.zeros((N, N_CANDIDATES), dtype=np.float32)
    all_cands  = None

    for s_idx, (seed, sw) in enumerate(zip(seeds, seed_weights)):
        models = seed_models[seed]  # 5 fold models
        seed_logits_list = []
        seed_cands_list  = []
        for start in tqdm(range(0, N, BATCH_SIZE),
                          desc=f"seed{seed}", leave=False):
            end = min(start + BATCH_SIZE, N)
            c   = torch.tensor(coords[start:end]).to(device)
            with torch.no_grad():
                cands_t  = make_candidates_gpu(c)
                seq_t    = make_seq_features_gpu(c)
                cand_t   = make_cand_features_gpu(c, cands_t)
                avg_l    = sum(m(seq_t, cand_t) for m in models) / len(models)
            seed_logits_list.append(avg_l.cpu().numpy())
            seed_cands_list.append(cands_t.cpu().numpy())

        all_logits += sw * np.concatenate(seed_logits_list, axis=0)
        if all_cands is None:
            all_cands = np.concatenate(seed_cands_list, axis=0)

    # Confidence → dynamic topk
    conf_dict = compute_confidence(all_logits, all_cands, temp=temp)
    combined  = conf_dict["combined"]
    topk_arr  = assign_topk(combined, t_low, t_high, k_low, k_normal, k_high)

    n_low    = int((topk_arr == k_high).sum())
    n_high   = int((topk_arr == k_low).sum())
    n_normal = N - n_low - n_high
    print(f"  topk 분포:  k={k_low} → {n_high:5d} ({n_high/N:.1%})  "
          f"k={k_normal} → {n_normal:5d} ({n_normal/N:.1%})  "
          f"k={k_high} → {n_low:5d} ({n_low/N:.1%})")

    # Predict
    logits_t = torch.tensor(all_logits).cpu()
    cands_t  = torch.tensor(all_cands).cpu()
    topk_t   = torch.tensor(topk_arr, dtype=torch.long)
    preds    = selector_predict_dynamic(logits_t, cands_t, topk_t, temp).numpy()

    # Save
    sub = pd.read_csv(SUBMISSION_PATH, index_col="id")
    df  = pd.DataFrame(preds, index=ids, columns=sub.columns)
    df.index.name = "id"
    if not out_name:
        seeds_str = "_".join(map(str, seeds))
        out_name  = (f"submission_dyntopk_s{seeds_str}_"
                     f"tl{int(t_low*100)}th{int(t_high*100)}_"
                     f"kl{k_low}kh{k_high}.csv")
    out_path = OUTPUT_DIR / out_name
    df.to_csv(out_path)
    print(f"  Saved: {out_path}  ({N} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Dynamic TopK inference / OOF calibration"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 777])
    parser.add_argument("--weights", type=float, nargs="+", default=None,
                        help="Seed weights (normalized). Default: equal.")
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP)
    parser.add_argument("--calibrate", action="store_true",
                        help="Run OOF calibration to find best thresholds")
    # Manual threshold overrides
    parser.add_argument("--t-low",   type=float, default=DEFAULT_T_LOW)
    parser.add_argument("--t-high",  type=float, default=DEFAULT_T_HIGH)
    parser.add_argument("--k-low",   type=int,   default=DEFAULT_K_LOW)
    parser.add_argument("--k-normal",type=int,   default=DEFAULT_K_NORMAL)
    parser.add_argument("--k-high",  type=int,   default=DEFAULT_K_HIGH)
    parser.add_argument("--out-name", type=str, default="",
                        help="Output CSV filename (default: auto-generated)")
    parser.add_argument("--no-submit", action="store_true",
                        help="Calibrate only, skip test submission")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Seeds: {args.seeds}")

    seed_models = load_selectors(args.seeds, device)

    if args.calibrate:
        print("\n=== OOF Calibration ===")
        train_ids, train_coords, train_labels = load_all(TRAIN_DIR, LABELS_PATH)
        best = calibrate(
            seed_models, train_ids, train_coords, train_labels, device,
            seed_weights=args.weights, temp=args.temp,
        )
        if args.no_submit:
            return
        # Use best config for submission
        t_low, t_high  = best["t_low"],  best["t_high"]
        k_low, k_high  = best["k_low"],  best["k_high"]
        k_normal       = DEFAULT_K_NORMAL
        print(f"\n  → Best config 적용하여 제출 파일 생성...")
    else:
        t_low, t_high  = args.t_low,   args.t_high
        k_low, k_high  = args.k_low,   args.k_high
        k_normal       = args.k_normal

    # Test inference
    test_ids, test_coords, _ = load_all(TEST_DIR)
    predict_test(
        seed_models, test_ids, test_coords, device,
        seed_weights=args.weights,
        t_low=t_low, t_high=t_high,
        k_low=k_low, k_normal=k_normal, k_high=k_high,
        temp=args.temp,
        out_name=args.out_name,
    )


if __name__ == "__main__":
    main()
