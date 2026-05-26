"""
C-group binary classifier (C vs non-C).
입력: export_oof_phase13.py 로 저장된 OOF data
출력:
  outputs/c_gate/c_gate_fold{i}.pt
  outputs/c_gate/c_gate_oof_probs.npy   (N,) OOF c_prob

Usage:
    python train_c_gate.py [--seed 42]
"""
import argparse, hashlib, sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from dataset import load_all
from features_phase13 import make_c_meta_features, N_META_FEATURES

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("  [WARNING] sklearn not found -- AUC metrics skipped")


OOF_DIR  = OUTPUT_DIR / "oof_phase13"
GATE_DIR = OUTPUT_DIR / "c_gate"


class CClassifier(nn.Module):
    def __init__(self, input_dim: int = N_META_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (B,) logit


def fold_id(sample_id, n_folds=N_FOLDS):
    return int(hashlib.md5(sample_id.encode()).hexdigest()[:8], 16) % n_folds


def r_hit(pred, true):
    return float(np.mean(np.linalg.norm(pred - true, axis=-1) <= R_HIT_THRESHOLD))


def threshold_sweep(c_prob_oof, c_labels, oof_preds, y_true, phys_alpha_col):
    print(f"\n{'thresh':>7}  {'prec':>6}  {'rec':>6}  {'pred_C%':>8}  {'base_hit_C':>11}  {'F1':>6}")
    print("-" * 60)
    best = {"thresh": 0.5, "score": 0.0}
    for t in np.arange(0.15, 0.90, 0.05):
        pred_c = c_prob_oof >= t
        if pred_c.sum() < 10:
            continue
        tp   = int((pred_c & c_labels).sum())
        prec = tp / pred_c.sum()
        rec  = tp / c_labels.sum()
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        ratio = pred_c.mean()
        base_hit_c = r_hit(oof_preds[pred_c], y_true[pred_c]) if pred_c.sum() > 0 else 0.0
        # score: reward precision >= 0.65, penalize ratio > 0.20
        score = f1 if prec >= 0.60 else 0.0
        marker = " <-- target" if (prec >= 0.65 and 0.08 <= ratio <= 0.22) else ""
        print(f"  {t:.2f}   {prec:.4f}  {rec:.4f}  {ratio:.1%}     {base_hit_c:.4f}       {f1:.4f}{marker}")
        if score > best["score"]:
            best = {
                "thresh": t, "score": score,
                "prec": prec, "rec": rec, "ratio": ratio,
            }
    # physics_routing comparison
    phys_mask = phys_alpha_col > 0
    if phys_mask.sum() > 0:
        phys_prec  = (phys_mask & c_labels).sum() / phys_mask.sum()
        phys_rec   = (phys_mask & c_labels).sum() / c_labels.sum()
        phys_ratio = phys_mask.mean()
        print(f"\n  [physics_routing 현재]  "
              f"prec={phys_prec:.4f}  rec={phys_rec:.4f}  ratio={phys_ratio:.1%}")
    print(f"\n  [권장 threshold]: {best['thresh']:.2f}  "
          f"prec={best.get('prec',0):.4f}  rec={best.get('rec',0):.4f}  "
          f"pred_C={best.get('ratio',0):.1%}")
    return best["thresh"]


def main(seed=SEED):
    # ── 데이터 로드 ───────────────────────────────────────────────
    ids, coords, _ = load_all(TRAIN_DIR, LABELS_PATH)
    N = len(ids)
    fold_ids = np.array([fold_id(i) for i in ids])

    for fname in ["oof_preds.npy", "oof_logits.npy", "oof_seq_feat.npy",
                  "oracle_cands.npy", "c_labels.npy", "y_true.npy"]:
        if not (OOF_DIR / fname).exists():
            raise FileNotFoundError(
                f"Missing: {OOF_DIR / fname}\n"
                f"  먼저 실행: python export_oof_phase13.py --seed {seed}"
            )

    oof_preds    = np.load(OOF_DIR / "oof_preds.npy")
    oof_logits   = np.load(OOF_DIR / "oof_logits.npy")
    oof_seq_feat = np.load(OOF_DIR / "oof_seq_feat.npy")
    oracle_cands = np.load(OOF_DIR / "oracle_cands.npy")
    c_labels     = np.load(OOF_DIR / "c_labels.npy")
    y_true       = np.load(OOF_DIR / "y_true.npy")

    n_c    = c_labels.sum()
    n_nc   = N - n_c
    print(f"N={N}  C={n_c}({n_c/N:.1%})  non-C={n_nc}({n_nc/N:.1%})")

    # meta features (25d, NO y_true)
    meta = make_c_meta_features(oof_seq_feat, oof_logits, oracle_cands, oof_preds, coords)
    print(f"Meta features: {meta.shape[1]}d")

    GATE_DIR.mkdir(parents=True, exist_ok=True)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c_float    = c_labels.astype(np.float32)
    pos_weight = torch.tensor([n_nc / n_c], device=device)
    print(f"pos_weight: {pos_weight.item():.2f}  |  device: {device}")

    oof_probs = np.zeros(N, dtype=np.float32)

    for fold in range(N_FOLDS):
        val_mask = fold_ids == fold
        trn_mask = ~val_mask

        X_trn = torch.tensor(meta[trn_mask]).to(device)
        y_trn = torch.tensor(c_float[trn_mask]).to(device)
        X_val = torch.tensor(meta[val_mask]).to(device)
        y_val = c_float[val_mask]

        model     = CClassifier(meta.shape[1]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        best_auc, best_state = 0.0, None
        patience_cnt = 0

        for epoch in range(150):
            model.train()
            perm = torch.randperm(len(X_trn))
            for start in range(0, len(X_trn), BATCH_SIZE):
                idx = perm[start:start + BATCH_SIZE]
                loss = criterion(model(X_trn[idx]), y_trn[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                probs_v = torch.sigmoid(model(X_val)).cpu().numpy()

            if HAS_SKLEARN:
                auc = roc_auc_score(y_val, probs_v)
            else:
                # simple proxy: mean prob of C-group vs non-C
                auc = float(probs_v[y_val == 1].mean() - probs_v[y_val == 0].mean() + 0.5)

            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                oof_probs[val_mask] = probs_v
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= 20:
                    break

        model.load_state_dict(best_state)
        torch.save(model.state_dict(), GATE_DIR / f"c_gate_fold{fold}.pt")

        val_probs = oof_probs[val_mask]
        if HAS_SKLEARN:
            fold_roc = roc_auc_score(y_val, val_probs)
            fold_pr  = average_precision_score(y_val, val_probs)
            print(f"  Fold {fold}: ROC-AUC={fold_roc:.4f}  PR-AUC={fold_pr:.4f}  "
                  f"best_auc={best_auc:.4f}")
        else:
            print(f"  Fold {fold}: best_proxy_auc={best_auc:.4f}")

    # ── 전체 OOF 평가 ────────────────────────────────────────────
    np.save(GATE_DIR / "c_gate_oof_probs.npy", oof_probs)

    print(f"\n{'='*55}")
    print(f"[전체 OOF]")
    if HAS_SKLEARN:
        roc = roc_auc_score(c_labels, oof_probs)
        pr  = average_precision_score(c_labels, oof_probs)
        print(f"  ROC-AUC : {roc:.4f}")
        print(f"  PR-AUC  : {pr:.4f}")

    # C-group에서 base selector 성능 (얼마나 어려운 그룹인지 확인)
    base_hit_c   = r_hit(oof_preds[c_labels],  y_true[c_labels])
    base_hit_nc  = r_hit(oof_preds[~c_labels], y_true[~c_labels])
    base_hit_all = r_hit(oof_preds, y_true)
    print(f"  Base R-Hit (all)   : {base_hit_all:.4f}")
    print(f"  Base R-Hit (C)     : {base_hit_c:.4f}")
    print(f"  Base R-Hit (non-C) : {base_hit_nc:.4f}")

    # threshold sweep
    phys_alpha_col = meta[:, 24]
    best_thresh = threshold_sweep(oof_probs, c_labels, oof_preds, y_true, phys_alpha_col)

    print(f"\n{'='*55}")
    if best_thresh > 0:
        print(f"[판정] CClassifier 학습 완료.")
        print(f"  권장 다음 명령:")
        print(f"    python train_residual_c.py --threshold {best_thresh:.2f}")
    else:
        print("[판정] precision 기준 미달 -- residual 구현 보류")

    print(f"\nSaved:")
    print(f"  {GATE_DIR}/c_gate_fold*.pt")
    print(f"  {GATE_DIR}/c_gate_oof_probs.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    main(parser.parse_args().seed)
