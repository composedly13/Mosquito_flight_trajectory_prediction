"""
Direct regression MLP: 11-timestep coords → predicted position.
Complements CandidateSelector via entropy-based blending at inference.

Training (single seed):
    python dev/experiments/regression.py --seed 42
    python dev/experiments/regression.py --seed 777

Outputs per seed:
    outputs/seed{N}/regmlp_fold{i}.pt   — fold models
    outputs/seed{N}/regmlp_oof.npy      — OOF coordinate predictions (N, 3)
"""
import argparse, hashlib, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    TRAIN_DIR, LABELS_PATH, OUTPUT_DIR,
    SEED, N_FOLDS, BATCH_SIZE, R_HIT_THRESHOLD,
)
from dataset import load_all
from candidates import make_seq_features_gpu

# Input: flattened seq features (11 timesteps × 11 features) + last point p0
_SEQ_FLAT = 11 * 11   # 121
_P0_DIM   = 3
REG_IN    = _SEQ_FLAT + _P0_DIM   # 124


class RegMLP(nn.Module):
    """
    Predicts mosquito position as p0 + learned offset.
    Input: seq_flat (B,121) + p0 (B,3).
    """
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(REG_IN, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 3),
        )

    def forward(self, seq_flat: torch.Tensor, p0: torch.Tensor) -> torch.Tensor:
        """seq_flat: (B,121)  p0: (B,3)  →  (B,3) predicted coords"""
        return p0 + self.net(torch.cat([seq_flat, p0], dim=-1))


def _fold_id(sample_id: str) -> int:
    return int(hashlib.md5(sample_id.encode()).hexdigest()[:8], 16) % N_FOLDS


def _make_features(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (seq_flat, p0) from (B,11,3) coords tensor."""
    seq_flat = make_seq_features_gpu(coords).flatten(1)   # (B, 121)
    p0       = coords[:, -1]                               # (B, 3)
    return seq_flat, p0


def _r_hit(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(pred - true, axis=-1) <= R_HIT_THRESHOLD))


# ── Training ───────────────────────────────────────────────────────────────────

def train_regression(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Seed: {seed}")

    ids, coords, labels = load_all(TRAIN_DIR, LABELS_PATH)
    N         = len(ids)
    fold_ids  = np.array([_fold_id(i) for i in ids])
    seed_dir  = OUTPUT_DIR / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    oof_preds = np.zeros((N, 3), dtype=np.float32)
    fold_hits = []

    for fold_idx in range(N_FOLDS):
        print(f"\n=== Fold {fold_idx+1}/{N_FOLDS} ===")
        val_mask  = fold_ids == fold_idx
        trn_mask  = ~val_mask
        trn_c, trn_l = coords[trn_mask], labels[trn_mask]
        val_c, val_l = coords[val_mask],  labels[val_mask]

        model     = RegMLP().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

        best_dist, best_state, wait = float("inf"), None, 0
        PATIENCE = 30

        bar = tqdm(range(150), ncols=100)
        for epoch in bar:
            # ── train ──────────────────────────────────────────
            model.train()
            perm = np.random.permutation(len(trn_c))
            for start in range(0, len(trn_c), BATCH_SIZE):
                idx = perm[start:start + BATCH_SIZE]
                c   = torch.tensor(trn_c[idx]).to(device)
                lbl = torch.tensor(trn_l[idx]).to(device)
                seq_flat, p0 = _make_features(c)
                loss = F.huber_loss(model(seq_flat, p0), lbl, delta=0.01)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

            # ── validate ───────────────────────────────────────
            model.eval()
            vp = []
            with torch.no_grad():
                for s in range(0, len(val_c), BATCH_SIZE):
                    c = torch.tensor(val_c[s:s + BATCH_SIZE]).to(device)
                    sf, p0 = _make_features(c)
                    vp.append(model(sf, p0).cpu().numpy())
            vp        = np.concatenate(vp)
            val_dist  = float(np.mean(np.linalg.norm(vp - val_l, axis=-1)))
            val_hit   = _r_hit(vp, val_l)
            bar.set_postfix(best=f"{best_dist*100:.3f}cm", hit=f"{val_hit:.4f}", patience=wait)

            if val_dist < best_dist:
                best_dist  = val_dist
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    bar.set_description(f"Early stop @ epoch {epoch+1}")
                    break

        model.load_state_dict(best_state)
        model.eval()
        vp = []
        with torch.no_grad():
            for s in range(0, len(val_c), BATCH_SIZE):
                c = torch.tensor(val_c[s:s + BATCH_SIZE]).to(device)
                sf, p0 = _make_features(c)
                vp.append(model(sf, p0).cpu().numpy())
        vp = np.concatenate(vp)
        hit = _r_hit(vp, val_l)
        print(f"\n  Fold {fold_idx+1} val R-Hit: {hit:.4f}  (best dist: {best_dist*100:.3f}cm)")
        fold_hits.append(hit)
        oof_preds[val_mask] = vp

        torch.save(model.state_dict(), seed_dir / f"regmlp_fold{fold_idx}.pt")

    oof_hit = _r_hit(oof_preds, labels)
    print(f"\n{'='*40}")
    print(f"CV mean R-Hit : {np.mean(fold_hits):.4f} ± {np.std(fold_hits):.4f}")
    print(f"OOF R-Hit     : {oof_hit:.4f}")
    np.save(seed_dir / "regmlp_oof.npy", oof_preds)
    print(f"Saved → {seed_dir}/regmlp_oof.npy")


# ── Inference helpers ──────────────────────────────────────────────────────────

def load_reg_models(seeds: list, device: torch.device) -> dict:
    """Returns {seed: [fold0_model, fold1_model, ...]} for all available folds."""
    all_models = {}
    for seed in seeds:
        seed_dir = OUTPUT_DIR / f"seed{seed}"
        fold_models = []
        for fold in range(N_FOLDS):
            path = seed_dir / f"regmlp_fold{fold}.pt"
            if path.exists():
                m = RegMLP().to(device)
                m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                m.eval()
                fold_models.append(m)
        if fold_models:
            all_models[seed] = fold_models
            print(f"  RegMLP seed{seed}: {len(fold_models)}/{N_FOLDS} folds 로드")
        else:
            print(f"  RegMLP seed{seed}: 없음 (python regression.py --seed {seed} 먼저 실행)")
    return all_models


def predict_reg_batch(
    reg_models: list,        # flat list of all loaded RegMLP models
    coords: torch.Tensor,    # (B, 11, 3)
) -> torch.Tensor:           # (B, 3)
    """Average predictions across all regression models."""
    with torch.no_grad():
        sf, p0 = _make_features(coords)
        return sum(m(sf, p0) for m in reg_models) / len(reg_models)


# ── Entropy-based blending ────────────────────────────────────────────────────

def entropy_blend(
    sel_preds:  np.ndarray,   # (N, 3)  selector top-k weighted avg
    reg_preds:  np.ndarray,   # (N, 3)  regression predictions
    sel_logits: np.ndarray,   # (N, C)  raw selector logits
    beta:       float = 1.0,  # blend strength: 0=pure selector, 1=full switch at H=1
) -> np.ndarray:              # (N, 3)
    """
    Alpha (selector weight) = clip(1 - beta * H, 0, 1)
    where H = normalised entropy of selector softmax ∈ [0, 1].
    H≈0 (confident)  → alpha≈1  → trust selector.
    H≈1 (uncertain)  → alpha≈(1-beta) → lean on regression.
    """
    logits_s = sel_logits - sel_logits.max(axis=1, keepdims=True)
    probs    = np.exp(logits_s)
    probs   /= probs.sum(axis=1, keepdims=True)
    C        = probs.shape[1]
    H        = -np.sum(probs * np.log(probs + 1e-9), axis=1) / np.log(C)   # (N,) ∈ [0,1]
    alpha    = np.clip(1.0 - beta * H, 0.0, 1.0)[:, np.newaxis]            # (N, 1)
    return alpha * sel_preds + (1.0 - alpha) * reg_preds


# ── TransformerRegressor inference helpers ────────────────────────────────────

def load_reg2_models(seeds: list, device: torch.device) -> dict:
    """Returns {seed: [fold0_model, ...]} for TransformerRegressor (reg2_fold{i}.pt)."""
    from model import TransformerRegressor  # lazy import to avoid circular at module load
    all_models = {}
    for seed in seeds:
        seed_dir = OUTPUT_DIR / f"seed{seed}"
        fold_models = []
        for fold in range(N_FOLDS):
            path = seed_dir / f"reg2_fold{fold}.pt"
            if path.exists():
                m = TransformerRegressor().to(device)
                m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                m.eval()
                fold_models.append(m)
        if fold_models:
            all_models[seed] = fold_models
            print(f"  Reg2(Transformer) seed{seed}: {len(fold_models)}/{N_FOLDS} folds 로드")
        else:
            print(f"  Reg2(Transformer) seed{seed}: 없음 (python train_regressor.py --seed {seed})")
    return all_models


def predict_reg2_batch(
    reg2_models: list,        # flat list of all loaded TransformerRegressor models
    coords: torch.Tensor,     # (B, 11, 3) on device
) -> torch.Tensor:            # (B, 3)
    """Average predictions across all TransformerRegressor models."""
    from candidates import make_seq_features_gpu
    with torch.no_grad():
        seq_f = make_seq_features_gpu(coords)   # (B, 11, 11)
        p0    = coords[:, -1, :]                # (B, 3) 마지막 알려진 위치
        return sum(m(seq_f, p0) for m in reg2_models) / len(reg2_models)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    train_regression(seed=args.seed)


# ── Physics-based routing (LSTM ensemble) ─────────────────────────────────────

def physics_routing_alpha(
    seq_feat: torch.Tensor,
    decel_thresh: float = 2.5,
    jerk_thresh:  float = 0.55,
    alpha_decel:  float = 0.30,
    alpha_jerk:   float = 0.30,
    max_alpha:    float = 0.30,
) -> torch.Tensor:
    """
    seq_feat의 last-timestep 신호로 C-group 가능성을 감지, per-sample LSTM 비중 결정.

    반환: (B,) alpha 값 — 0.0(selector-only) ~ max_alpha(강한 LSTM 비중)

    신호:
      speed_ratio (feat idx 1): prev_speed / current_speed
        > decel_thresh → 급감속 → C-group 극감속 클러스터 (latency_s075 nearest 25%)
        실측값: 일반 ~1.0, 극감속 2.0+
      jerk_abs (feat idx 9): ||acc-prev_acc|| / 0.05m
        > jerk_thresh → 강한 jerk → C-group jerk 클러스터 (jerk_xl nearest 25%)
        실측값: 일반 ~0.04, C-group jerk ~0.1-0.3

    임계값 캘리브레이션:
      decel_thresh=1.4: 현재 speed가 이전 step의 70% 이하 (40% 감속)
      jerk_thresh=0.25: jerk_vec = 12.5mm/step (전형 5mm 대비 2.5×)
      max_alpha=0.45: 완전 LSTM 신뢰 금지 (selector가 더 강한 영역 보호)

    C-group 분포 (Phase 11):
      극감속  25.2%: latency_s075 nearest → decel_thresh 커버
      극jerk  25.1%: jerk_xl nearest      → jerk_thresh 커버
      극회전  22.0%: turn_p/n060 nearest  → 별도 신호 없음 (미커버)

    조정: 학습 후 predict.py 출력 'Routing 발동' 비율로 임계값 튜닝 가능.
    목표 발동률: 10-15% (C-group 극감속/jerk ≈ 12.5%)
    """
    # Last timestep features (t=10 기준)
    speed_ratio = seq_feat[:, -1, 1]   # prev_speed / current_speed
    jerk_abs    = seq_feat[:, -1, 9]   # ||jerk|| / 0.05

    decel_signal = (speed_ratio > decel_thresh).float() * alpha_decel
    jerk_signal  = (jerk_abs    > jerk_thresh).float()  * alpha_jerk

    return (decel_signal + jerk_signal).clamp(0.0, max_alpha)


# ── MosquitoLSTM inference helpers ────────────────────────────────────────────

def load_lstm_models(seeds: list, device: torch.device) -> list:
    """
    lstm_fold{i}.pt 파일을 로드해 flat list로 반환.
    train_lstm.py --seed {seed} 로 학습 후 사용.
    """
    from model import MosquitoLSTM
    models = []
    for seed in seeds:
        seed_dir = OUTPUT_DIR / f"seed{seed}"
        for fold in range(N_FOLDS):
            path = seed_dir / f"lstm_fold{fold}.pt"
            if path.exists():
                m = MosquitoLSTM().to(device)
                m.load_state_dict(
                    torch.load(path, map_location=device, weights_only=True)
                )
                m.eval()
                models.append(m)
    if models:
        print(f"  LSTM: {len(models)} models 로드 ({len(seeds)} seeds)")
    else:
        print("  LSTM: 없음 — python dev/experiments/train_lstm.py --seed 42 먼저 실행")
    return models


def predict_lstm_batch(
    lstm_models: list,
    coords:      torch.Tensor,   # (B, 11, 3)
    seq_feat:    torch.Tensor,   # (B, 11, 11) — 이미 계산된 seq_features 재사용
) -> torch.Tensor:               # (B, 3)
    """
    모든 LSTM 모델의 예측을 평균. seq_feat는 predict 루프에서 이미 계산된 것을 전달.
    """
    with torch.no_grad():
        return sum(m(seq_feat, coords) for m in lstm_models) / len(lstm_models)
