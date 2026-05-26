"""
C-classifier meta feature 생성 함수 (y_true 독립).

make_c_meta_features() 를 train_c_gate.py 와 predict_phase13.py 양쪽에서 임포트.
반환: (N, 25) float32 -- y_true 의존 없음, leakage 없음.

Feature index 설명:
  [0..10]  last-step seq features (t=10)
  [11..15] last 3 steps (t=8,9,10) aggregation
  [16..19] logit-based (fold OOF logits 사용 -- fold-aware로 관리 필요)
  [20..22] candidate spread
  [23..24] physics divergence
"""
import numpy as np
from config import EPS, R_HIT_THRESHOLD
from candidates import N_CANDIDATES


FEATURE_NAMES = [
    # [0..10] last-step seq features (t=10)
    "speed_last",          # seq_feat[10,0]: speed / 0.05
    "prev_speed_ratio",    # seq_feat[10,1]: prev_speed / cur_speed (decel)
    "acc_norm",            # seq_feat[10,2]: acc_norm / speed
    "acc_par",             # seq_feat[10,3]: parallel accel / speed
    "acc_perp",            # seq_feat[10,4]: perp accel / speed (lateral turn)
    "jerk_norm",           # seq_feat[10,5]
    "turn_cos",            # seq_feat[10,6]: direction consistency
    "curvature",           # seq_feat[10,7]: curvature * 10
    "direction_flag",      # seq_feat[10,8]: sign(acc_par)
    "jerk_abs",            # seq_feat[10,9]: ||jerk|| / 0.05  (C-group key signal)
    "acc_cos",             # seq_feat[10,10]: accel direction consistency
    # [11..15] last 3 steps aggregation
    "speed_ratio_max3",    # max decel in last 3 steps
    "jerk_abs_max3",       # max jerk in last 3 steps
    "turn_cos_mean3",      # mean linearity in last 3 steps
    "acc_perp_mean3",      # mean lateral accel in last 3 steps
    "acc_cos_min3",        # min direction consistency in last 3 steps
    # [16..19] logit-based (fold OOF, leakage-free)
    "entropy_H",           # selector uncertainty
    "logit_gap",           # top1 - top2 logit
    "top1_prob",           # softmax top1 probability
    "top5_prob_sum",       # cumulative top5 probability
    # [20..22] candidate spread
    "cand_pos_std",        # std of all 52 candidate positions
    "topk_pos_std",        # std of top10 candidates by logit
    "cand_centroid_dist",  # ||base_pred - mean(cands)|| / speed_h
    # [23..24] physics divergence
    "linear_vs_accel",     # ||linear_pred - accel_pred|| / speed_h
    "physics_alpha",       # physics_routing_alpha value (0.0 or >0)
]

N_META_FEATURES = len(FEATURE_NAMES)  # 25


def make_c_meta_features(
    seq_feat:  np.ndarray,   # (N, 11, 11) -- from make_seq_features
    logits:    np.ndarray,   # (N, C)      -- OOF fold-based logits, NO y_true
    cands:     np.ndarray,   # (N, C, 3)   -- oracle candidates
    base_pred: np.ndarray,   # (N, 3)      -- OOF fold-based predictions
    coords:    np.ndarray,   # (N, 11, 3)  -- raw trajectory
) -> np.ndarray:             # (N, 25) float32 -- NO y_true dependency
    N = len(seq_feat)

    # [0..10] last-step seq features
    last = seq_feat[:, 10, :]   # (N, 11)

    # [11..15] last 3 steps (t=8,9,10)
    last3 = seq_feat[:, 8:11, :]                              # (N, 3, 11)
    speed_ratio_max3 = last3[:, :, 1].max(axis=1)[:, None]
    jerk_abs_max3    = last3[:, :, 9].max(axis=1)[:, None]
    turn_cos_mean3   = last3[:, :, 6].mean(axis=1)[:, None]
    acc_perp_mean3   = last3[:, :, 4].mean(axis=1)[:, None]
    acc_cos_min3     = last3[:, :, 10].min(axis=1)[:, None]

    # [16..19] logit-based (fold OOF)
    logits_s = logits - logits.max(axis=1, keepdims=True)
    probs    = np.exp(logits_s)
    probs   /= probs.sum(axis=1, keepdims=True)

    H         = (-np.sum(probs * np.log(probs + 1e-9), axis=1) / np.log(N_CANDIDATES))[:, None]
    sorted_l  = np.sort(logits, axis=1)[:, ::-1]
    logit_gap = (sorted_l[:, 0] - sorted_l[:, 1])[:, None]
    top1_prob = probs.max(axis=1, keepdims=True)
    top5_prob = np.sort(probs, axis=1)[:, -5:].sum(axis=1, keepdims=True)

    # [20..22] candidate spread
    cand_pos_std = cands.std(axis=1).mean(axis=1, keepdims=True)   # (N, 1)

    top10_idx   = np.argsort(-logits, axis=1)[:, :10]              # (N, 10)
    top10_cands = cands[np.arange(N)[:, None], top10_idx]          # (N, 10, 3)
    topk_pos_std = top10_cands.std(axis=1).mean(axis=1, keepdims=True)

    p0      = coords[:, 10]                                        # (N, 3)
    d1      = coords[:, 10] - coords[:, 9]
    speed_h = np.linalg.norm(d1, axis=1, keepdims=True).clip(EPS) * 2.0
    centroid = cands.mean(axis=1)                                  # (N, 3)
    cand_centroid_dist = (
        np.linalg.norm(base_pred - centroid, axis=1, keepdims=True) / speed_h
    )

    # [23..24] physics divergence
    linear_pred      = p0 + 2.0 * d1
    d2               = coords[:, 9] - coords[:, 8]
    acc              = d1 - d2
    accel_pred       = p0 + 2.0 * d1 + acc
    linear_vs_accel  = (
        np.linalg.norm(linear_pred - accel_pred, axis=1, keepdims=True) / speed_h
    )

    # physics_routing_alpha 재현 (외부 의존 없음)
    speed_ratio_last = seq_feat[:, -1, 1:2]
    jerk_abs_last    = seq_feat[:, -1, 9:10]
    physics_alpha    = (
        (speed_ratio_last > 2.5).astype(np.float32) * 0.30
        + (jerk_abs_last  > 0.55).astype(np.float32) * 0.30
    ).clip(0.0, 0.30)

    feat = np.concatenate([
        last,               # 11
        speed_ratio_max3,   # 1
        jerk_abs_max3,      # 1
        turn_cos_mean3,     # 1
        acc_perp_mean3,     # 1
        acc_cos_min3,       # 1
        H,                  # 1
        logit_gap,          # 1
        top1_prob,          # 1
        top5_prob,          # 1
        cand_pos_std,       # 1
        topk_pos_std,       # 1
        cand_centroid_dist, # 1
        linear_vs_accel,    # 1
        physics_alpha,      # 1
    ], axis=1).astype(np.float32)   # (N, 25)

    assert feat.shape == (N, N_META_FEATURES), f"shape mismatch: {feat.shape}"
    return feat
