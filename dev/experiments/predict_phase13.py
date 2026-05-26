"""
Phase 13 최종 inference: selector/GCN + C-gate + GRU residual → submission CSV.

Usage:
    python predict_phase13.py --seeds 42 777 123 --threshold 0.70 --alpha 1.0

구조:
    base_pred = multi-seed selector ensemble
    c_prob    = CClassifier(meta_features)
    if c_prob > threshold:
        delta      = GRUResidual(seq_feat, base_pred - p0)
        final_pred = base_pred + alpha * delta
    else:
        final_pred = base_pred
"""
import argparse, sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    TEST_DIR, SUBMISSION_PATH, OUTPUT_DIR,
    SEED, N_FOLDS, BATCH_SIZE, TOPK,
)
from dataset import load_all
from model import CandidateSelector, selector_predict
from candidates import (
    make_candidates_gpu, make_seq_features_gpu, make_cand_features_gpu, N_CANDIDATES,
)
from features_phase13 import make_c_meta_features
from train_c_gate import CClassifier
from train_residual_c import GRUResidual

GATE_DIR  = OUTPUT_DIR / "c_gate"
RESID_DIR = OUTPUT_DIR / "residual_c"
CORRECTION_CAP = 0.006


def load_selectors(seeds, device):
    models = []
    for seed in seeds:
        for fold in range(N_FOLDS):
            path = OUTPUT_DIR / f"seed{seed}" / f"selector_fold{fold}.pt"
            if not path.exists():
                raise FileNotFoundError(f"Missing: {path}")
            m = CandidateSelector().to(device)
            m.load_state_dict(torch.load(path, map_location=device), strict=False)
            m.eval()
            models.append(m)
    print(f"  {len(models)} selector models ({len(seeds)} seeds × {N_FOLDS} folds)")
    return models


def load_c_gate(device):
    models = []
    for fold in range(N_FOLDS):
        path = GATE_DIR / f"c_gate_fold{fold}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}\n  먼저: python train_c_gate.py")
        m = CClassifier().to(device)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        m.eval()
        models.append(m)
    print(f"  {len(models)} C-gate models loaded")
    return models


def load_residuals(device):
    models = []
    for fold in range(N_FOLDS):
        path = RESID_DIR / f"residual_fold{fold}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}\n  먼저: python train_residual_c.py")
        m = GRUResidual().to(device)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        m.eval()
        models.append(m)
    print(f"  {len(models)} residual models loaded")
    return models


def predict(seeds=None, threshold=0.70, alpha=1.0, temp=2.0):
    if seeds is None:
        seeds = [SEED]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Seeds: {seeds}  |  threshold={threshold}  alpha={alpha}")

    selectors  = load_selectors(seeds, device)
    gate_models = load_c_gate(device)
    resid_models = load_residuals(device)

    ids, coords, _ = load_all(TEST_DIR)
    N = len(ids)
    print(f"Test samples: {N}")

    all_preds    = []
    all_logits   = []
    all_seq_feat = []
    all_cands    = []

    for start in tqdm(range(0, N, BATCH_SIZE), desc="Base inference"):
        end  = min(start + BATCH_SIZE, N)
        c    = torch.tensor(coords[start:end]).to(device)

        with torch.no_grad():
            cands_t    = make_candidates_gpu(c)
            seq_t      = make_seq_features_gpu(c)
            cand_t     = make_cand_features_gpu(c, cands_t)
            avg_logits = sum(m(seq_t, cand_t, cands_t) for m in selectors) / len(selectors)
            pred       = selector_predict(avg_logits, cands_t, topk=TOPK, temp=temp)

        all_preds.append(pred.cpu().numpy())
        all_logits.append(avg_logits.cpu().numpy())
        all_seq_feat.append(seq_t.cpu().numpy())
        all_cands.append(cands_t.cpu().numpy())

    all_preds    = np.concatenate(all_preds,    axis=0)   # (N, 3)
    all_logits   = np.concatenate(all_logits,   axis=0)   # (N, C)
    all_seq_feat = np.concatenate(all_seq_feat, axis=0)   # (N, 11, 11)
    all_cands    = np.concatenate(all_cands,    axis=0)   # (N, C, 3)

    # ── C-gate: meta features → c_prob ──────────────────────────
    p0_all        = coords[:, 10, :]                       # (N, 3)
    meta          = make_c_meta_features(
        all_seq_feat, all_logits, all_cands, all_preds, coords
    )
    meta_t        = torch.tensor(meta).to(device)

    with torch.no_grad():
        # 5-fold average (test에는 OOF 없으므로 모든 fold 평균)
        c_logits = sum(m(meta_t) for m in gate_models) / len(gate_models)
        c_prob   = torch.sigmoid(c_logits).cpu().numpy()   # (N,)

    pred_c_mask = c_prob >= threshold
    n_routed    = int(pred_c_mask.sum())
    print(f"\nC-gate routing: {n_routed} / {N} ({n_routed/N*100:.1f}%)")

    # ── Residual correction: C-gate 통과 샘플만 ─────────────────
    final_preds = all_preds.copy()

    if n_routed > 0:
        c_idx       = np.where(pred_c_mask)[0]
        seq_c       = torch.tensor(all_seq_feat[c_idx]).to(device)
        base_delta_c = torch.tensor(all_preds[c_idx] - p0_all[c_idx]).to(device)

        with torch.no_grad():
            avg_delta = sum(m(seq_c, base_delta_c) for m in resid_models) / len(resid_models)
        delta_np = avg_delta.cpu().numpy()                  # (n_routed, 3)

        final_preds[c_idx] = all_preds[c_idx] + alpha * delta_np
        print(f"  delta norm: mean={np.linalg.norm(delta_np, axis=1).mean()*100:.2f}cm  "
              f"max={np.linalg.norm(delta_np, axis=1).max()*100:.2f}cm")

    # ── 저장 ─────────────────────────────────────────────────────
    sub = pd.read_csv(SUBMISSION_PATH, index_col="id")

    def save_csv(preds, name):
        df = pd.DataFrame(preds, index=ids, columns=sub.columns)
        df.index.name = "id"
        path = OUTPUT_DIR / name
        df.to_csv(path)
        print(f"  Saved: {path}")

    seeds_str = "_".join(map(str, seeds))
    save_csv(all_preds,    f"submission_phase13_base_{seeds_str}.csv")
    save_csv(final_preds,  f"submission_phase13_blend_{seeds_str}_t{threshold:.2f}_a{alpha:.1f}.csv")
    print(f"\n제출 파일: submission_phase13_blend_{seeds_str}_t{threshold:.2f}_a{alpha:.1f}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",     type=int, nargs="+", default=[SEED])
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--alpha",     type=float, default=1.0)
    parser.add_argument("--temp",      type=float, default=2.0)
    args = parser.parse_args()
    predict(seeds=args.seeds, threshold=args.threshold, alpha=args.alpha, temp=args.temp)