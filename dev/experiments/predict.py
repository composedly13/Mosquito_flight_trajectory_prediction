"""
Inference: K-Fold selector ensemble + boundary MLP correction -> submission CSV.
"""
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import (
    TEST_DIR, SUBMISSION_PATH, OUTPUT_DIR,
    SEED, N_FOLDS, BATCH_SIZE,
)
from dataset import load_all
from model import CandidateSelector, selector_predict
from candidates import make_candidates_gpu, make_seq_features_gpu, make_cand_features_gpu
from boundary import BoundaryMLP, apply_boundary


def load_selectors(device: torch.device) -> list:
    models = []
    for fold in range(N_FOLDS):
        path = OUTPUT_DIR / f"selector_fold{fold}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        m = CandidateSelector().to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        models.append(m)
    return models


def load_boundary(device: torch.device):
    path = OUTPUT_DIR / "boundary.pt"
    if not path.exists():
        print("No boundary model found -- skipping correction.")
        return None
    m = BoundaryMLP().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


def predict_batch(
    selectors: list,
    seq_feat:  torch.Tensor,   # (B, 11, 9)
    cand_feat: torch.Tensor,   # (B, C, 10)
    cands:     torch.Tensor,   # (B, C, 3)
) -> np.ndarray:               # (B, 3)
    with torch.no_grad():
        avg_logits = sum(m(seq_feat, cand_feat) for m in selectors) / len(selectors)
        pred = selector_predict(avg_logits, cands)
    return pred.cpu().numpy()


def predict():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    selectors = load_selectors(device)
    boundary  = load_boundary(device)

    ids, coords, _ = load_all(TEST_DIR)
    N = len(ids)
    print(f"Test samples: {N}")

    all_preds  = []
    all_coords = []

    for start in tqdm(range(0, N, BATCH_SIZE), desc="Inference"):
        end  = min(start + BATCH_SIZE, N)
        c_np = coords[start:end]                           # (B, 11, 3) numpy
        c    = torch.tensor(c_np).to(device)

        with torch.no_grad():
            cands_t = make_candidates_gpu(c)
            seq_t   = make_seq_features_gpu(c)
            cand_t  = make_cand_features_gpu(c, cands_t)

        pred = predict_batch(selectors, seq_t, cand_t, cands_t)
        all_preds.append(pred)
        all_coords.append(c_np)

    all_preds  = np.concatenate(all_preds,  axis=0)   # (N, 3)
    all_coords = np.concatenate(all_coords, axis=0)   # (N, 11, 3)

    all_preds = apply_boundary(boundary, all_coords, all_preds, device)

    sub = pd.read_csv(SUBMISSION_PATH, index_col="id")
    df  = pd.DataFrame(all_preds, index=ids, columns=sub.columns)
    df.index.name = "id"

    out_path = OUTPUT_DIR / "submission.csv"
    df.to_csv(out_path)
    print(f"Submission saved: {out_path}  ({N} rows)")


if __name__ == "__main__":
    predict()
