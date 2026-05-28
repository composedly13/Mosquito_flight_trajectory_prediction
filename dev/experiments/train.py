import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from tqdm import tqdm
import hashlib

from config import *  # includes TOPK, LISTMLE_WEIGHT, PAIRWISE_WEIGHT, SOFT_TEMP
from dataset import (
    load_all, MosquitoDataset,
    augment_batch_gpu, augment_batch_gpu_yaw, augment_speed_scale_gpu,
    augment_mirror_gpu, augment_noise_gpu,
)
from model import CandidateSelector, soft_labels, selector_predict
from candidates import (N_CANDIDATES, N_CANDIDATES_BASE, N_EXTRA_V1,
                        make_candidates, make_candidates_gpu,
                        make_seq_features_gpu, make_cand_features_gpu,
                        compute_gate_mask_gpu,
                        CANDIDATE_FAMILY, FAMILY_NAMES)


def r_hit(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(pred - true, axis=-1) <= R_HIT_THRESHOLD))


def fold_id(sample_id: str, n_folds: int = N_FOLDS) -> int:
    return int(hashlib.md5(sample_id.encode()).hexdigest()[:8], 16) % n_folds


def soft_ce_loss(logits: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    """Soft cross-entropy: 거리 기반 soft label 분포와 CE."""
    return -(soft * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


# ── Stage 2-A: turn-aware soft label reweight ──────────────────────────────
_TURN_MASK: torch.Tensor | None = None
_JERK_MASK: torch.Tensor | None = None

def _get_family_masks(device: torch.device) -> tuple:
    global _TURN_MASK, _JERK_MASK
    if _TURN_MASK is None or _TURN_MASK.device != device:
        turn_fid = list(FAMILY_NAMES).index("turn")
        jerk_fid = list(FAMILY_NAMES).index("jerk")
        _TURN_MASK = torch.tensor(CANDIDATE_FAMILY == turn_fid,
                                  dtype=torch.float32, device=device)
        _JERK_MASK = torch.tensor(CANDIDATE_FAMILY == jerk_fid,
                                  dtype=torch.float32, device=device)
    return _TURN_MASK, _JERK_MASK


def turn_aware_soft_labels(
    soft:        torch.Tensor,   # (B, C)
    cands:       torch.Tensor,   # (B, C, 3)
    true:        torch.Tensor,   # (B, 3)
    turn_boost:  float = 1.10,
    jerk_decay:  float = 1.00,
) -> torch.Tensor:
    """
    oracle family가 turn인 샘플의 soft label에서
    turn 후보를 turn_boost배, jerk 후보를 jerk_decay배 조정 후 재정규화.
    oracle이 없는 샘플(C-group)은 변경하지 않음.
    """
    if turn_boost == 1.00 and jerk_decay == 1.00:
        return soft

    turn_m, jerk_m = _get_family_masks(soft.device)

    # Find oracle per sample
    dist        = (cands - true.unsqueeze(1)).norm(dim=-1)  # (B, C)
    oracle_dist = dist.min(dim=-1).values                    # (B,)
    oracle_idx  = dist.argmin(dim=-1)                        # (B,)

    # Oracle family
    fam_t = torch.tensor(CANDIDATE_FAMILY, device=soft.device)
    turn_fid = list(FAMILY_NAMES).index("turn")
    is_turn_oracle = ((fam_t[oracle_idx] == turn_fid) &
                      (oracle_dist <= R_HIT_THRESHOLD))      # (B,)

    # Weight modifier: only for turn-oracle samples
    wmod = torch.ones_like(soft)                             # (B, C)
    if turn_boost != 1.00:
        wmod[is_turn_oracle] = torch.where(
            turn_m.bool().unsqueeze(0).expand(is_turn_oracle.sum(), -1),
            wmod[is_turn_oracle] * turn_boost,
            wmod[is_turn_oracle],
        )
    if jerk_decay != 1.00:
        wmod[is_turn_oracle] = torch.where(
            jerk_m.bool().unsqueeze(0).expand(is_turn_oracle.sum(), -1),
            wmod[is_turn_oracle] * jerk_decay,
            wmod[is_turn_oracle],
        )

    soft_mod = soft * wmod
    soft_mod = soft_mod / (soft_mod.sum(dim=-1, keepdim=True) + 1e-8)
    return soft_mod


def pairwise_loss(logits: torch.Tensor, soft: torch.Tensor, margin: float = 0.12) -> torch.Tensor:
    """Ranking loss: good candidates should score higher than bad ones."""
    good = (soft > 0.1).float()
    bad  = (soft < 0.01).float()
    score_good = (logits * good).sum(dim=-1) / (good.sum(dim=-1) + 1e-8)
    score_bad  = (logits * bad).sum(dim=-1)  / (bad.sum(dim=-1)  + 1e-8)
    return F.relu(margin - score_good + score_bad).mean()


def listmle_loss(logits: torch.Tensor, cands: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Oracle 후보를 top-1으로 직접 최적화: -log P(oracle ranked first)."""
    dist       = torch.norm(cands - true.unsqueeze(1), dim=-1)  # (B, C)
    oracle_idx = dist.argmin(dim=-1)                             # (B,)
    return -F.log_softmax(logits, dim=-1).gather(1, oracle_idx.unsqueeze(1)).mean()


def focal_listmle_loss(
    logits: torch.Tensor,
    cands:  torch.Tensor,
    true:   torch.Tensor,
    gamma:  float = 1.0,
    mode:   str   = "oracle_rank",
) -> torch.Tensor:
    """
    Focal ListMLE: oracle 후보가 상위권에 없는 샘플에 더 큰 가중치 부여.

    mode="oracle_rank":
      - oracle 후보의 현재 rank를 구하고, rank가 높을수록(순위가 낮을수록) 가중치 증가
      - weight = (oracle_rank / C) ^ gamma
      - B-group 샘플이 자연히 높은 가중치를 받음

    mode="margin":
      - oracle logit과 top-1 logit의 gap이 클수록 가중치 증가
      - weight = sigmoid(gap) ^ gamma

    C-group (oracle dist > 1cm) 샘플은 loss=0 (정답 후보 없음 → 학습 불가)
    """
    dist        = torch.norm(cands - true.unsqueeze(1), dim=-1)  # (B, C)
    oracle_dist = dist.min(dim=-1).values                          # (B,)
    oracle_idx  = dist.argmin(dim=-1)                              # (B,)

    # oracle log-prob (ListMLE base term)
    log_probs   = F.log_softmax(logits, dim=-1)                    # (B, C)
    oracle_logp = log_probs.gather(1, oracle_idx.unsqueeze(1)).squeeze(1)  # (B,)

    if mode == "oracle_rank":
        # oracle의 현재 rank (0=top1, C-1=last)
        with torch.no_grad():
            ranks = (logits > logits.gather(1, oracle_idx.unsqueeze(1))).sum(dim=-1).float()
        weight = (ranks / logits.size(1)) ** gamma                 # (B,)
    elif mode == "margin":
        with torch.no_grad():
            top1_logit    = logits.max(dim=-1).values
            oracle_logit  = logits.gather(1, oracle_idx.unsqueeze(1)).squeeze(1)
            gap = (top1_logit - oracle_logit).clamp(min=0.0)
        weight = torch.sigmoid(gap) ** gamma                       # (B,)
    else:
        raise ValueError(f"Unknown focal mode: {mode}")

    # C-group 마스킹
    is_ab = (oracle_dist <= R_HIT_THRESHOLD).float()               # (B,)
    loss  = -(weight * oracle_logp * is_ab)
    return loss.mean()


def oracle_margin_loss(logits: torch.Tensor, cands: torch.Tensor, true: torch.Tensor,
                       k: int = 5, margin: float = 0.15) -> torch.Tensor:
    """
    Oracle Top-K Margin Loss:
    oracle 후보의 logit이 k번째 높은 logit보다 margin만큼 높아야 한다.
    → oracle이 top-k 안에 들어오도록 직접 강제.
    oracle이 없는 샘플(C-group)은 loss=0.

    k=5: oracle을 top-5로 끌어올림 (현재 Oracle in Top-5 = 39.2% → 개선 목표)
    margin=0.15: 충분한 마진 확보
    """
    dist        = torch.norm(cands - true.unsqueeze(1), dim=-1)   # (B, C)
    oracle_dist = dist.min(dim=-1).values                          # (B,)
    oracle_idx  = dist.argmin(dim=-1)                              # (B,)

    oracle_score = logits.gather(1, oracle_idx.unsqueeze(1))       # (B, 1)
    # k번째로 높은 logit: oracle이 이것보다 높아야 top-k에 들어감
    kth_score    = logits.kthvalue(logits.size(1) - k + 1, dim=1).values.unsqueeze(1)  # (B, 1)

    loss = F.relu(kth_score - oracle_score + margin)               # (B, 1)

    # C-group(oracle 없음)은 학습에서 제외
    is_hit = (oracle_dist <= R_HIT_THRESHOLD).float().unsqueeze(1) # (B, 1)
    return (loss * is_hit).mean()


def train_fold(
    fold: int,
    ids: list,
    coords: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    patience: int = PATIENCE,
    cand_dim: int | None = None,
):
    val_mask   = np.array([fold_id(i) == fold for i in ids])
    train_mask = ~val_mask

    train_ds = MosquitoDataset(coords[train_mask], labels[train_mask], augment=True)
    val_ds   = MosquitoDataset(coords[val_mask],   labels[val_mask],   augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model     = CandidateSelector(cand_dim=cand_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_hit, best_state, patience_cnt = 0.0, None, 0
    best_preds = np.zeros((val_mask.sum(), 3), dtype=np.float32)

    pbar = tqdm(range(1, EPOCHS + 1), desc=f"Fold {fold}")
    for epoch in pbar:
        # Train
        model.train()
        for batch in train_loader:
            coords_b = batch["coords"].to(device, non_blocking=True)
            true     = batch["label"].to(device, non_blocking=True)

            if AUG_MODE == 'so3':
                coords_b, true = augment_batch_gpu(coords_b, true)
            elif AUG_MODE == 'yaw':
                coords_b, true = augment_batch_gpu_yaw(coords_b, true)
            elif AUG_MODE == 'yaw_speed':
                coords_b, true = augment_batch_gpu_yaw(coords_b, true)
                coords_b, true = augment_speed_scale_gpu(
                    coords_b, true,
                    scale_range=SPEED_SCALE_RANGE,
                    prob=SPEED_SCALE_PROB,
                )

            if AUG_FLIP:
                coords_b, true = augment_mirror_gpu(coords_b, true)
            if AUG_NOISE:
                coords_b = augment_noise_gpu(coords_b, std=NOISE_STD)

            cands  = make_candidates_gpu(coords_b)
            seq_f  = make_seq_features_gpu(coords_b)
            cand_f = make_cand_features_gpu(coords_b, cands)

            logits = model(seq_f, cand_f)                               # (B, C)
            soft   = soft_labels(cands, true)                           # (B, C)
            if TURN_TARGET_BOOST != 1.00 or JERK_TARGET_DECAY != 1.00:
                soft = turn_aware_soft_labels(soft, cands, true,
                                              TURN_TARGET_BOOST, JERK_TARGET_DECAY)

            # Stage 2-B: gate mask — inactive extra candidates → masked out
            if C_GATE_V1_ENABLED:
                gate_mask = compute_gate_mask_gpu(coords_b)             # (B, C)
                logits = logits.masked_fill(~gate_mask, -1e9)
                soft   = soft * gate_mask.float()
                soft   = soft / (soft.sum(-1, keepdim=True) + 1e-8)

            loss = soft_ce_loss(logits, soft) + PAIRWISE_WEIGHT * pairwise_loss(logits, soft)
            if LISTMLE_WEIGHT > 0:
                loss = loss + LISTMLE_WEIGHT * listmle_loss(logits, cands, true)
            if FOCAL_LML_WEIGHT > 0:
                loss = loss + FOCAL_LML_WEIGHT * focal_listmle_loss(
                    logits, cands, true,
                    gamma=FOCAL_LML_GAMMA,
                    mode=FOCAL_LML_MODE,
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # Val
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                coords_b = batch["coords"].to(device, non_blocking=True)

                cands  = make_candidates_gpu(coords_b)
                seq_f  = make_seq_features_gpu(coords_b)
                cand_f = make_cand_features_gpu(coords_b, cands)

                logits = model(seq_f, cand_f)
                pred   = selector_predict(logits, cands, topk=TOPK)
                preds.append(pred.cpu().numpy())
                trues.append(batch["label"].cpu().numpy())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        hit   = r_hit(preds, trues)

        if hit > best_hit:
            best_hit   = hit
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_preds = preds.copy()
            patience_cnt = 0
        else:
            patience_cnt += 1

        pbar.set_postfix(hit=f"{hit:.4f}", best=f"{best_hit:.4f}", patience=patience_cnt)

        if patience_cnt >= patience:
            print(f"  Early stop at epoch {epoch}")
            break

    return best_hit, best_state, val_mask, best_preds


def train(seed: int = SEED, patience: int = PATIENCE,
          listmle_weight: float = LISTMLE_WEIGHT,
          out_tag: str = "",
          sign_feat: bool = False,
          turn_boost: float = TURN_TARGET_BOOST,
          jerk_decay: float = JERK_TARGET_DECAY):
    global LISTMLE_WEIGHT, TURN_TARGET_BOOST, JERK_TARGET_DECAY
    LISTMLE_WEIGHT    = listmle_weight
    TURN_TARGET_BOOST = turn_boost
    JERK_TARGET_DECAY = jerk_decay

    # Stage 2-A: sign feature flag
    import config as _cfg
    import candidates as _cands_mod
    _cfg.CAND_FEAT_SIGN   = sign_feat
    _cands_mod.CAND_FEAT_SIGN = sign_feat
    # CAND_DIM for model construction
    from model import CAND_DIM as _base_cand_dim
    _active_cand_dim = (_base_cand_dim
                        if not sign_feat
                        else _base_cand_dim + 2)  # +2 sign features

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Candidates: {N_CANDIDATES}  |  Seed: {seed}"
          f"  |  Aug: {AUG_MODE}  |  Patience: {patience}"
          f"  |  LML={listmle_weight}"
          f"  |  CAND_DIM={_active_cand_dim}"
          f"  |  turn_boost={turn_boost:.2f}  jerk_decay={jerk_decay:.2f}")

    # Each seed gets its own subdirectory so multi-seed runs don't overwrite each other.
    out_dir = OUTPUT_DIR / f"seed{seed}{out_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids, coords, labels = load_all(TRAIN_DIR, LABELS_PATH)

    fold_results = []
    all_states   = []
    oof_preds    = np.zeros((len(ids), 3), dtype=np.float32)

    for fold in range(N_FOLDS):
        print(f"\n=== Fold {fold + 1}/{N_FOLDS} ===")
        hit, state, val_mask, val_preds = train_fold(
            fold, ids, coords, labels, device,
            patience=patience, cand_dim=_active_cand_dim
        )
        fold_results.append(hit)
        all_states.append(state)
        oof_preds[val_mask] = val_preds
        print(f"Fold {fold + 1} best R-Hit: {hit:.4f}")

    print(f"\n{'='*40}")
    print(f"CV mean R-Hit: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    for i, h in enumerate(fold_results):
        print(f"  Fold {i+1}: {h:.4f}")

    oof_hit = r_hit(oof_preds, labels)
    print(f"OOF R-Hit (selector only): {oof_hit:.4f}")

    # Oracle: 후보군 상한선 진단
    oracle_cands     = make_candidates(coords)                         # (N, C, 3)
    min_dists        = np.linalg.norm(
        oracle_cands - labels[:, np.newaxis, :], axis=-1
    ).min(axis=1)
    oracle_hit_score = float(np.mean(min_dists <= R_HIT_THRESHOLD))
    print(f"Oracle R-Hit ({N_CANDIDATES} candidates): {oracle_hit_score:.4f}"
          f"  (selector efficiency: {oof_hit / oracle_hit_score:.1%})")

    # Save selector models
    for i, state in enumerate(all_states):
        torch.save(state, out_dir / f"selector_fold{i}.pt")

    print(f"\nModels saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--listmle-weight", type=float, default=LISTMLE_WEIGHT)
    parser.add_argument("--out-tag", type=str, default="",
                        help="Suffix for model dir (e.g. '_stage2a_sign')")
    # Stage 2-A flags
    parser.add_argument("--sign-feat", action="store_true",
                        help="Enable jerk/perp sign features (CAND_DIM 14→16)")
    parser.add_argument("--turn-boost", type=float, default=TURN_TARGET_BOOST)
    parser.add_argument("--jerk-decay", type=float, default=JERK_TARGET_DECAY)
    # Stage 2-B flag
    parser.add_argument("--gate-v1", action="store_true",
                        help="Gated C-candidates v1 (Smart50+6, JT_q95 gate)")
    args = parser.parse_args()

    # Stage 2-B: extend CANDIDATES before training starts
    if args.gate_v1:
        import config as _cfg2b
        import candidates as _cands2b
        _cfg2b.C_GATE_V1_ENABLED   = True
        _cands2b.C_GATE_V1_ENABLED = True
        from candidates import (CANDIDATES as _bc, _EXTRA_CANDIDATES_V1 as _ev1,
                                 N_CANDIDATES_BASE as _nb, _family_id as _fid)
        if len(_cands2b.CANDIDATES) == _nb:   # extend only once
            _cands2b.CANDIDATES   = list(_bc) + list(_ev1)
            _cands2b.N_CANDIDATES = len(_cands2b.CANDIDATES)
            import numpy as _np
            _cands2b.CANDIDATE_FAMILY = _np.array(
                [_fid(s.name) for s in _cands2b.CANDIDATES], dtype=_np.int64
            )
            _cands2b._CAND_PARAMS_CACHE.clear()

    train(seed=args.seed, patience=args.patience,
          listmle_weight=args.listmle_weight, out_tag=args.out_tag,
          sign_feat=args.sign_feat,
          turn_boost=args.turn_boost, jerk_decay=args.jerk_decay)
