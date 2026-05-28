"""
Stage 1 bias 검증 스크립트.
Seed-별 / 2-seed-ensemble OOF에서 perp+turn bias 효과 확인.

Usage:
  python eval_bias.py               # seed42 + seed777 검증
  python eval_bias.py --seed 42     # single seed
"""
from __future__ import annotations
import argparse, hashlib
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import TRAIN_DIR, LABELS_PATH, OUTPUT_DIR, N_FOLDS, BATCH_SIZE
from dataset import load_all
from model import CandidateSelector
from candidates import (
    make_candidates, make_candidates_gpu, make_seq_features_gpu,
    make_cand_features_gpu, make_cand_features,
    N_CANDIDATES, motion_terms, EPS,
)


def fold_id(sid: str) -> int:
    return int(hashlib.md5(sid.encode()).hexdigest()[:8], 16) % N_FOLDS


def r_hit(p, t):
    return float(np.mean(np.linalg.norm(p - t, axis=-1) <= 0.01))


def get_oof_logits(models, ids, coords, device):
    N = len(ids)
    lg = np.full((N, N_CANDIDATES), -np.inf, np.float32)
    for fi, m in models.items():
        m.eval()
        vm = np.array([fold_id(i) == fi for i in ids])
        vc = coords[vm]; vi = np.where(vm)[0]
        for s in range(0, len(vc), BATCH_SIZE):
            e = min(s + BATCH_SIZE, len(vc))
            c = torch.tensor(vc[s:e]).to(device)
            with torch.no_grad():
                ct = make_candidates_gpu(c)
                st = make_seq_features_gpu(c)
                ft = make_cand_features_gpu(c, ct)
                lg[vi[s:e]] = m(st, ft).cpu().numpy()
    return lg


def topk_pred(logits, cands_np, topk, temp):
    N, C, _ = cands_np.shape
    probs = np.exp(logits / temp - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    si = np.argsort(-probs, axis=1)
    sp = probs[np.arange(N)[:, None], si][:, :topk]
    sc = cands_np[np.arange(N)[:, None], si][:, :topk]
    sp /= sp.sum(axis=1, keepdims=True) + 1e-9
    return (sc * sp[:, :, None]).sum(axis=1)


def compute_bias(coords, cands_np, logits, w_perp=0.06, w_turn=0.02):
    """Return bias matrix (N, C) for B-risk samples."""
    # Interaction features
    cf = make_cand_features(coords, cands_np)          # (N, C, 14)
    perp_match = cf[:, :, 12]                          # (N, C)
    jerk_match  = cf[:, :, 13]                         # (N, C)

    # Physics for gates
    p0_, d1_, d2_, acc_, jk_v = motion_terms(coords, end_idx=10)
    sp_  = np.linalg.norm(d1_, axis=1, keepdims=True).clip(EPS)
    tg_  = d1_ / sp_
    aps_ = (acc_ * tg_).sum(axis=1, keepdims=True)
    apv_ = acc_ - aps_ * tg_
    acc_perp_n = np.linalg.norm(apv_, axis=1) / (sp_[:, 0] + EPS)
    jerk_n_    = np.linalg.norm(jk_v,  axis=1) / (sp_[:, 0] + EPS)

    # Confidence features for B-risk gate
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    sl = np.sort(logits, axis=1)[:, ::-1]
    sp2 = np.sort(probs, axis=1)[:, ::-1]
    gap  = sl[:, 0] - sl[:, 1]
    p5   = sp2[:, :5].copy(); p5 /= p5.sum(1, keepdims=True) + 1e-9
    ent5 = -(p5 * np.log(p5 + 1e-9)).sum(1) / np.log(5)

    q40g  = np.percentile(gap,          40)
    q60e  = np.percentile(ent5,         60)
    q70p  = np.percentile(acc_perp_n,   70)
    q70j  = np.percentile(jerk_n_,      70)

    b_risk   = (gap <= q40g) | (ent5 >= q60e) | (acc_perp_n >= q70p) | (jerk_n_ >= q70j)
    turn_gate = acc_perp_n >= q70p

    bias = np.zeros((len(coords), N_CANDIDATES), np.float32)
    bias[b_risk] = (w_perp * perp_match + w_turn * perp_match * turn_gate[:, None])[b_risk]
    return bias, b_risk.sum()


def eval_seed(seed, ids, coords, labels, device, topk=10, temp=1.0,
              w_perp=0.06, w_turn=0.02):
    """OOF evaluation with and without bias for a single seed."""
    seed_dir = OUTPUT_DIR / f"seed{seed}"
    models = {}
    for fi in range(N_FOLDS):
        p = seed_dir / f"selector_fold{fi}.pt"
        if not p.exists():
            raise FileNotFoundError(p)
        m = CandidateSelector().to(device)
        m.load_state_dict(torch.load(p, map_location=device), strict=False)
        m.eval()
        models[fi] = m

    cands_np = make_candidates(coords)
    logits   = get_oof_logits(models, ids, coords, device)

    # Baseline
    base_preds = topk_pred(logits, cands_np, topk, temp)
    base_hit   = r_hit(base_preds, labels)

    # With bias
    bias, n_risk = compute_bias(coords, cands_np, logits, w_perp, w_turn)
    biased_logits = logits + bias
    bias_preds = topk_pred(biased_logits, cands_np, topk, temp)
    bias_hit   = r_hit(bias_preds, labels)

    # A/B/C groups
    dist_c = np.linalg.norm(cands_np - labels[:, None, :], axis=-1)
    od = dist_c.min(1); oi = dist_c.argmin(1)
    sli = np.argsort(-logits, axis=1)
    orank = (sli == oi[:, None]).argmax(1)
    is_hit = od <= 0.01; in_t5 = orank < 5
    ga = is_hit & in_t5; gb = is_hit & ~in_t5; gc = ~is_hit

    bh_a = r_hit(base_preds[ga], labels[ga])
    bh_b = r_hit(base_preds[gb], labels[gb])
    bah_a = r_hit(bias_preds[ga], labels[ga])
    bah_b = r_hit(bias_preds[gb], labels[gb])

    print(f"  seed{seed}  topk={topk}  temp={temp}  w_perp={w_perp}  w_turn={w_turn}")
    print(f"  B-risk: {n_risk}/{len(ids)} ({n_risk/len(ids):.1%})")
    print(f"  Baseline : OOF={base_hit:.4f}  A={bh_a:.4f}  B={bh_b:.4f}")
    print(f"  +bias    : OOF={bias_hit:.4f}  A={bah_a:.4f}  B={bah_b:.4f}"
          f"  (dOOF={bias_hit-base_hit:+.4f})")
    return base_hit, bias_hit, logits, biased_logits, cands_np


def eval_ensemble(ids, coords, labels, device,
                  seeds=(42, 777), weights=(0.5, 0.5),
                  topk=10, temp=1.0, w_perp=0.06, w_turn=0.02):
    """
    2-seed ensemble OOF with and without bias.
    Tests two strategies:
      A) bias applied AFTER averaging (from ensemble logits)
      B) bias applied BEFORE averaging (per-seed logits, then average)
    """
    print(f"\n  [2-seed ensemble]  seeds={list(seeds)}  weights={list(weights)}")
    cands_np = make_candidates(coords)
    N = len(ids)

    # Collect per-seed OOF logits
    per_seed_logits = {}
    for seed, w in zip(seeds, weights):
        seed_dir = OUTPUT_DIR / f"seed{seed}"
        models = {}
        for fi in range(N_FOLDS):
            p = seed_dir / f"selector_fold{fi}.pt"
            m = CandidateSelector().to(device)
            m.load_state_dict(torch.load(p, map_location=device), strict=False)
            m.eval()
            models[fi] = m
        per_seed_logits[seed] = (w, get_oof_logits(models, ids, coords, device))

    logits_sum = sum(w * lg for (w, lg) in per_seed_logits.values())

    base_preds = topk_pred(logits_sum, cands_np, topk, temp)
    base_hit   = r_hit(base_preds, labels)

    # Strategy A: bias from ensemble logits
    bias_a, n_risk_a = compute_bias(coords, cands_np, logits_sum, w_perp, w_turn)
    biased_a = logits_sum + bias_a
    bias_preds_a = topk_pred(biased_a, cands_np, topk, temp)
    bias_hit_a   = r_hit(bias_preds_a, labels)

    # Strategy B: bias applied per-seed, then averaged
    biased_sum = np.zeros((N, N_CANDIDATES), np.float32)
    n_risk_b = 0
    for seed, (w, lg) in per_seed_logits.items():
        bias_s, nr = compute_bias(coords, cands_np, lg, w_perp, w_turn)
        biased_sum += w * (lg + bias_s)
        n_risk_b = nr  # same physics gate, approx same

    bias_preds_b = topk_pred(biased_sum, cands_np, topk, temp)
    bias_hit_b   = r_hit(bias_preds_b, labels)

    # Use strategy B result as "biased" for the rest
    biased = biased_sum
    bias_preds = bias_preds_b
    bias_hit   = bias_hit_b

    dist_c = np.linalg.norm(cands_np - labels[:, None, :], axis=-1)
    od = dist_c.min(1); oi = dist_c.argmin(1)
    sli = np.argsort(-logits_sum, axis=1)
    orank = (sli == oi[:, None]).argmax(1)
    ga = (od <= 0.01) & (orank < 5)
    gb = (od <= 0.01) & (orank >= 5)

    print(f"  Baseline    : OOF={base_hit:.4f}  A={r_hit(base_preds[ga],labels[ga]):.4f}"
          f"  B={r_hit(base_preds[gb],labels[gb]):.4f}")
    print(f"  +bias(A-avg): OOF={bias_hit_a:.4f}  A={r_hit(bias_preds_a[ga],labels[ga]):.4f}"
          f"  B={r_hit(bias_preds_a[gb],labels[gb]):.4f}"
          f"  (dOOF={bias_hit_a-base_hit:+.4f})  [bias after avg]")
    print(f"  +bias(B-per): OOF={bias_hit_b:.4f}  A={r_hit(bias_preds_b[ga],labels[ga]):.4f}"
          f"  B={r_hit(bias_preds_b[gb],labels[gb]):.4f}"
          f"  (dOOF={bias_hit_b-base_hit:+.4f})  [bias per-seed, then avg]")
    return base_hit, bias_hit_a, bias_hit_b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 777])
    parser.add_argument("--topk",  type=int, default=10)
    parser.add_argument("--temp",  type=float, default=1.0)
    parser.add_argument("--w-perp", type=float, default=0.06)
    parser.add_argument("--w-turn", type=float, default=0.02)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ids, coords, labels = load_all(TRAIN_DIR, LABELS_PATH)
    print(f"Loaded {len(ids)} samples\n")

    print("=" * 50)
    print("Per-seed OOF validation")
    print("=" * 50)
    for seed in args.seeds:
        eval_seed(seed, ids, coords, labels, device,
                  topk=args.topk, temp=args.temp,
                  w_perp=args.w_perp, w_turn=args.w_turn)
        print()

    if len(args.seeds) > 1:
        print("=" * 50)
        print("2-seed ensemble OOF")
        print("=" * 50)
        eval_ensemble(ids, coords, labels, device,
                      seeds=tuple(args.seeds), weights=(0.5, 0.5),
                      topk=args.topk, temp=args.temp,
                      w_perp=args.w_perp, w_turn=args.w_turn)


if __name__ == "__main__":
    main()
