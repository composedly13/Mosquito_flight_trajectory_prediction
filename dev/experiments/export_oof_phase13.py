"""
현재 best selector 모델의 OOF data를 outputs/oof_phase13/ 에 저장.
multi-seed 지원: 각 fold에서 모든 seed의 logits를 평균해 OOF 생성
→ predict_phase13.py의 multi-seed ensemble과 동일한 분포

저장 파일:
  oof_preds.npy       (N, 3)    -- fold-based OOF predictions (multi-seed avg)
  oof_logits.npy      (N, C)    -- fold-based OOF logits (multi-seed avg)
  oof_seq_feat.npy    (N,11,11) -- seq features (y_true 독립)
  oracle_cands.npy    (N, C, 3) -- 52 candidates
  c_labels.npy        (N,)      -- bool, oracle_min_dist > 0.01
  oracle_min_dist.npy (N,)      -- float, 분석용
  y_true.npy          (N, 3)    -- training labels
  sample_ids.npy      (N,)      -- str array

Usage:
    python export_oof_phase13.py --seeds 42 777 123
    python export_oof_phase13.py --seed 42          # single seed
"""
import argparse, hashlib, sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from dataset import load_all
from model import CandidateSelector, selector_predict
from candidates import (
    make_candidates, make_candidates_gpu,
    make_seq_features_gpu, make_cand_features_gpu, N_CANDIDATES,
)


def fold_id(sample_id, n_folds=N_FOLDS):
    return int(hashlib.md5(sample_id.encode()).hexdigest()[:8], 16) % n_folds


def main(seeds):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Seeds: {seeds}")

    ids, coords, labels = load_all(TRAIN_DIR, LABELS_PATH)
    N = len(ids)
    fold_ids = np.array([fold_id(i) for i in ids])

    # 각 seed × fold 모델 로드
    # seed_fold_models[seed][fold] = model
    seed_fold_models = {}
    for seed in seeds:
        fold_models = []
        for f in range(N_FOLDS):
            path = OUTPUT_DIR / f"seed{seed}" / f"selector_fold{f}.pt"
            if not path.exists():
                raise FileNotFoundError(f"Missing: {path}  (run: python train.py --seed {seed})")
            m = CandidateSelector().to(device)
            m.load_state_dict(torch.load(path, map_location=device), strict=False)
            m.eval()
            fold_models.append(m)
        seed_fold_models[seed] = fold_models
        print(f"  seed{seed}: {N_FOLDS} fold models loaded")

    oof_preds    = np.zeros((N, 3),            dtype=np.float32)
    oof_logits   = np.zeros((N, N_CANDIDATES), dtype=np.float32)
    oof_seq_feat = np.zeros((N, 11, 11),       dtype=np.float32)

    for start in tqdm(range(0, N, BATCH_SIZE), desc="OOF export"):
        end = min(start + BATCH_SIZE, N)
        c   = torch.tensor(coords[start:end]).to(device)
        fi  = fold_ids[start:end]

        with torch.no_grad():
            cands_t = make_candidates_gpu(c)
            seq_t   = make_seq_features_gpu(c)
            cand_t  = make_cand_features_gpu(c, cands_t)

        oof_seq_feat[start:end] = seq_t.cpu().numpy()

        for f in range(N_FOLDS):
            mask = (fi == f)
            if not mask.any():
                continue
            idx = np.where(mask)[0]

            # 모든 seed의 logits 평균 (multi-seed ensemble)
            with torch.no_grad():
                avg_lgts = sum(
                    seed_fold_models[s][f](seq_t[mask], cand_t[mask], cands_t[mask])
                    for s in seeds
                ) / len(seeds)
                pred = selector_predict(avg_lgts, cands_t[mask], topk=TOPK, temp=2.0)

            oof_logits[start + idx] = avg_lgts.cpu().numpy()
            oof_preds [start + idx] = pred.cpu().numpy()

    # C-label (y_true는 training split에서만 사용)
    print("Computing oracle candidates...")
    oracle_cands = make_candidates(coords)
    min_dists    = np.linalg.norm(
        oracle_cands - labels[:, np.newaxis, :], axis=-1
    ).min(axis=1)
    c_labels = (min_dists > R_HIT_THRESHOLD)

    out = OUTPUT_DIR / "oof_phase13"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "oof_preds.npy",        oof_preds)
    np.save(out / "oof_logits.npy",       oof_logits)
    np.save(out / "oof_seq_feat.npy",     oof_seq_feat)
    np.save(out / "oracle_cands.npy",     oracle_cands)
    np.save(out / "c_labels.npy",         c_labels)
    np.save(out / "oracle_min_dist.npy",  min_dists)
    np.save(out / "y_true.npy",           labels)
    np.save(out / "sample_ids.npy",       np.array(ids))

    oof_hit = float(np.mean(np.linalg.norm(oof_preds - labels, axis=-1) <= R_HIT_THRESHOLD))
    c_ratio = c_labels.mean()
    near_c  = ((min_dists > R_HIT_THRESHOLD) & (min_dists <= 0.015)).mean()
    hard_c  = (min_dists > 0.015).mean()

    print(f"\n[OOF Summary ({len(seeds)} seeds)]")
    print(f"  OOF R-Hit      : {oof_hit:.4f}")
    print(f"  C-group ratio  : {c_ratio:.1%}  ({c_labels.sum()} / {N})")
    print(f"  near-C (1~1.5cm): {near_c:.1%}")
    print(f"  hard-C (>1.5cm) : {hard_c:.1%}")
    print(f"  Saved -> {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--seed",  type=int, default=None)
    group.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()
    if args.seeds:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [SEED]
    main(seeds)
