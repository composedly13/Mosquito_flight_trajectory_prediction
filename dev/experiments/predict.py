"""
Inference: K-Fold selector ensemble + optional multi-seed ensemble -> submission CSV.
Optional entropy-based regression blend when RegMLP models are available.

Single seed:   python predict.py --seed 42
Multi-seed:    python predict.py --seeds 42 777
With blend:    python predict.py --seeds 42 777  --beta 1.0
"""
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import (
    TEST_DIR, SUBMISSION_PATH, OUTPUT_DIR,
    SEED, N_FOLDS, BATCH_SIZE, TOPK,
)
from dataset import load_all
from model import CandidateSelector, selector_predict
from candidates import make_candidates_gpu, make_seq_features_gpu, make_cand_features_gpu
from boundary import BoundaryMLP, apply_boundary
from regression import load_reg_models, predict_reg_batch, entropy_blend


def load_selectors(seeds: list, device: torch.device) -> list:
    """Load all fold models for the given seeds.  seeds=[42] for single-seed inference."""
    models = []
    for seed in seeds:
        seed_dir = OUTPUT_DIR / f"seed{seed}"
        for fold in range(N_FOLDS):
            path = seed_dir / f"selector_fold{fold}.pt"
            if not path.exists():
                raise FileNotFoundError(f"Missing: {path}  (run: python train.py --seed {seed})")
            m = CandidateSelector().to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            models.append(m)
    print(f"  {len(models)} selector models loaded ({len(seeds)} seeds × {N_FOLDS} folds)")
    return models


def load_boundary(device: torch.device):
    path = OUTPUT_DIR / "boundary.pt"
    if not path.exists():
        return None
    m = BoundaryMLP().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


def predict(seeds: list = None, beta: float = 1.0):
    if seeds is None:
        seeds = [SEED]
    torch.manual_seed(seeds[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Seeds: {seeds}")

    selectors = load_selectors(seeds, device)
    boundary  = load_boundary(device)

    # Load regression models if available (for entropy-blend submission)
    print("  RegMLP 모델 확인...")
    reg_seed_models = load_reg_models(seeds, device)
    reg_models_flat = [m for fold_list in reg_seed_models.values() for m in fold_list]
    use_blend = len(reg_models_flat) > 0

    ids, coords, _ = load_all(TEST_DIR)
    N = len(ids)
    print(f"Test samples: {N}")

    all_preds   = []
    all_logits  = []
    all_reg     = [] if use_blend else None

    for start in tqdm(range(0, N, BATCH_SIZE), desc="Inference"):
        end  = min(start + BATCH_SIZE, N)
        c_np = coords[start:end]
        c    = torch.tensor(c_np).to(device)

        with torch.no_grad():
            cands_t = make_candidates_gpu(c)
            seq_t   = make_seq_features_gpu(c)
            cand_t  = make_cand_features_gpu(c, cands_t)
            avg_logits = sum(m(seq_t, cand_t) for m in selectors) / len(selectors)
            pred       = selector_predict(avg_logits, cands_t, topk=TOPK)

        all_preds.append(pred.cpu().numpy())
        all_logits.append(avg_logits.cpu().numpy())

        if use_blend:
            reg_pred = predict_reg_batch(reg_models_flat, c)
            all_reg.append(reg_pred.cpu().numpy())

    all_preds  = np.concatenate(all_preds,  axis=0)   # (N, 3)
    all_logits = np.concatenate(all_logits, axis=0)   # (N, C)
    if use_blend:
        all_reg = np.concatenate(all_reg, axis=0)      # (N, 3)

    sub = pd.read_csv(SUBMISSION_PATH, index_col="id")

    def save_csv(preds: np.ndarray, name: str):
        df = pd.DataFrame(preds, index=ids, columns=sub.columns)
        df.index.name = "id"
        path = OUTPUT_DIR / name
        df.to_csv(path)
        print(f"  {path}  ({N} rows)")

    print("\n[제출 파일 생성]")

    # 1. Selector-only (권장 baseline)
    save_csv(all_preds, "submission.csv")

    # 2. Entropy-blend (RegMLP 있을 때)
    if use_blend:
        blended = entropy_blend(all_preds, all_reg, all_logits, beta=beta)
        save_csv(blended, "submission_blend.csv")
        print(f"  ※ entropy blend β={beta}  (β 조정: --beta 값)")

    # 3. Boundary (이전 실험 잔재, 피처 불일치 시 skip)
    if boundary is not None:
        try:
            all_coords = np.concatenate([
                coords[s:min(s + BATCH_SIZE, N)]
                for s in range(0, N, BATCH_SIZE)
            ], axis=0)
            corrected = apply_boundary(boundary, all_coords, all_preds, device)
            save_csv(corrected, "submission_boundary.csv")
        except RuntimeError as e:
            print(f"  boundary skip (피처 불일치): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--seed",  type=int, default=None,
                       help="Single seed (default: config.SEED)")
    group.add_argument("--seeds", type=int, nargs="+", default=None,
                       help="Multiple seeds, e.g. --seeds 42 777")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Entropy-blend strength: 0=pure selector, 1=full switch (default: 1.0)")
    args = parser.parse_args()

    if args.seeds:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [SEED]

    predict(seeds=seeds, beta=args.beta)
