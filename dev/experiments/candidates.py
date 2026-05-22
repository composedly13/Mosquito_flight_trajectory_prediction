"""
Frenet-frame physics candidate generation.
Each candidate is a physically plausible prediction of position at +80ms.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from config import EPS


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    d1: float           # velocity scale (2.0 = pure linear extrapolation)
    par: float          # acceleration parallel component scale
    perp: float         # acceleration perpendicular component scale
    d2: float = 0.0     # previous velocity scale
    jerk: float = 0.0   # jerk scale
    time_scale: float = 1.0  # latency correction


# Smart 50-cand (2026-05-22): 10개 제거 + 10개 교체, 총 50개 유지
#
# 제거: acc_2d1_040/050 (weak acc), jerk_small±0.08 (xl/l로 커버됨),
#       frenet_par090/100_p000 + par085/095_n005 (frenet_best 중복 4개),
#       latency_s088/l112 (s085~s092, l108~l115 사이 보간 중복)
#
# 추가: latency_s075/s080/l120 (C-group nearest 21.8% = latency_s085 → 더 강한 보정),
#       turn_p070/n070/p080/n080 (C-group |perp|>0.60 15.5%, |perp| p95=1.02),
#       jerk_xxl±0.80 (C-group jerk p75=0.55, p95=1.70),
#       frenet_par130_n020 (C-group par>1.20 20.4%)
CANDIDATES = [
    # Base
    CandidateSpec("p0_2d1",            2.00,  0.00,  0.00),

    # Acceleration family (강한 것만 유지)
    CandidateSpec("acc_2d1_056",       1.98,  0.56,  0.56),
    CandidateSpec("acc_2d1_060",       2.00,  0.60,  0.60),

    # Frenet family (핵심 — 중복 4개 제거)
    CandidateSpec("frenet_best",       1.98,  0.96, -0.08),
    CandidateSpec("frenet_par100_n010",2.00,  1.00, -0.10),
    CandidateSpec("frenet_par090_p020",1.96,  0.90,  0.20),
    CandidateSpec("frenet_par080_p020",2.02,  0.80,  0.20),

    # Turn family
    CandidateSpec("frenet_par110_n020",1.94,  1.10, -0.20),
    CandidateSpec("frenet_fast_p100",  2.06,  1.00, -0.08),
    CandidateSpec("frenet_slow_p100",  1.90,  1.00, -0.08),
    CandidateSpec("frenet_par070_n020",1.98,  0.70, -0.20),
    CandidateSpec("frenet_par120_n020",1.98,  1.20, -0.20),
    CandidateSpec("frenet_par120_p020",1.98,  1.20,  0.20),
    CandidateSpec("frenet_fast_p120_n020", 2.08, 1.20, -0.20),
    CandidateSpec("frenet_slow_p070_p020", 1.86, 0.70,  0.20),
    CandidateSpec("frenet_par130_n020",1.92,  1.30, -0.20),  # ★NEW: C-group par>1.20 20.4%

    # Jerk family (small 제거, xxl 추가)
    CandidateSpec("jerk_med_pos",      1.98,  0.90, -0.05, jerk= 0.15),
    CandidateSpec("jerk_med_neg",      1.98,  0.90, -0.05, jerk=-0.15),
    CandidateSpec("jerk_l_pos",        1.98,  0.88,  0.00, jerk= 0.30),
    CandidateSpec("jerk_l_neg",        1.98,  0.88,  0.00, jerk=-0.30),
    CandidateSpec("jerk_xl_pos",       1.96,  0.82,  0.00, jerk= 0.50),
    CandidateSpec("jerk_xl_neg",       1.96,  0.82,  0.00, jerk=-0.50),
    CandidateSpec("jerk_xxl_pos",      1.94,  0.76,  0.00, jerk= 0.80),  # ★NEW: C-group jerk
    CandidateSpec("jerk_xxl_neg",      1.94,  0.76,  0.00, jerk=-0.80),  # ★NEW

    # Latency family (s088/l112 제거, s075/s080/l120 추가)
    CandidateSpec("latency_s075",      1.98,  0.96, -0.08, time_scale=0.75),  # ★NEW
    CandidateSpec("latency_s080",      1.98,  0.96, -0.08, time_scale=0.80),  # ★NEW: C-group 21.8%
    CandidateSpec("latency_s085",      1.98,  0.96, -0.08, time_scale=0.85),
    CandidateSpec("latency_s092",      1.98,  0.96, -0.08, time_scale=0.92),
    CandidateSpec("latency_l108",      1.98,  0.96, -0.08, time_scale=1.08),
    CandidateSpec("latency_l115",      1.98,  0.96, -0.08, time_scale=1.15),
    CandidateSpec("latency_l120",      1.98,  0.96, -0.08, time_scale=1.20),  # ★NEW
    CandidateSpec("latency_l110_turn", 1.98,  1.10, -0.20, time_scale=1.10),
    CandidateSpec("latency_s090_turn", 1.96,  0.90,  0.20, time_scale=0.90),

    # 추가 Frenet
    CandidateSpec("frenet_par105_n010",2.00,  1.05, -0.10),
    CandidateSpec("frenet_par075_p010",1.98,  0.75,  0.10),

    # 급격한 방향 전환 (p070/n070/p080/n080 추가, p030/n030 유지)
    CandidateSpec("turn_p030",         2.00,  0.80,  0.30),
    CandidateSpec("turn_n030",         2.00,  0.80, -0.30),
    CandidateSpec("turn_p045",         1.96,  0.70,  0.45),
    CandidateSpec("turn_n045",         1.96,  0.70, -0.45),
    CandidateSpec("turn_p060",         1.90,  0.55,  0.60),
    CandidateSpec("turn_n060",         1.90,  0.55, -0.60),
    CandidateSpec("turn_p070",         1.84,  0.45,  0.70),  # ★NEW: C-group |perp|>0.60 15.5%
    CandidateSpec("turn_n070",         1.84,  0.45, -0.70),  # ★NEW
    CandidateSpec("turn_p080",         1.78,  0.38,  0.80),  # ★NEW: C-group |perp| p95=1.02
    CandidateSpec("turn_n080",         1.78,  0.38, -0.80),  # ★NEW

    # Turn + Jerk 복합
    CandidateSpec("tj_pp20",           1.98,  0.85,  0.20, jerk= 0.20),
    CandidateSpec("tj_np20",           1.98,  0.85, -0.20, jerk= 0.20),
    CandidateSpec("tj_pn20",           1.98,  0.85,  0.20, jerk=-0.20),
    CandidateSpec("tj_nn20",           1.98,  0.85, -0.20, jerk=-0.20),
    CandidateSpec("turn_fast_n030",    2.08,  0.80, -0.30),

]
# 60-cand 실험(2026-05-22): jerk_xxl~turn_n100 10개 *추가* → oracle +3.4pp, efficiency -4.9pp → 실패
# Smart 50-cand: 10개 *교체* (총 수 50 유지) + 더 극단 후보(jerk>0.80, perp>0.80) 미포함
# 성공 기준: oracle ≥ 76%, efficiency ≥ 86%, OOF ≥ 0.652

N_CANDIDATES = len(CANDIDATES)

FAMILY_NAMES = ["base", "acc", "frenet", "turn", "jerk", "latency"]

def _family_id(name: str) -> int:
    if name == "p0_2d1":              return 0
    if name.startswith("acc_"):       return 1
    if name.startswith("latency"):    return 5
    if "jerk" in name:                return 4
    if any(x in name for x in ["fast", "slow", "par070", "par120", "par110",
                                 "par075", "par105", "turn_", "tj_"]):
        return 3
    return 2

CANDIDATE_FAMILY = np.array([_family_id(s.name) for s in CANDIDATES], dtype=np.int64)


def motion_terms(x: np.ndarray, end_idx: int = -1):
    """p0, d1(velocity), acc(acceleration), prev_acc, jerk"""
    idx = end_idx if end_idx >= 0 else x.shape[1] + end_idx
    p0       = x[:, idx]
    d1       = x[:, idx]     - x[:, idx - 1]
    d2       = x[:, idx - 1] - x[:, idx - 2]
    acc      = d1 - d2
    prev_acc = d2 - (x[:, idx - 2] - x[:, idx - 3])
    jerk     = acc - prev_acc
    return p0, d1, d2, acc, jerk


def make_candidates(x: np.ndarray) -> np.ndarray:
    """
    x: (N, 11, 3)
    returns: (N, C, 3) candidate positions
    """
    p0, d1, d2, acc, jerk = motion_terms(x, end_idx=10)

    speed    = np.linalg.norm(d1, axis=1, keepdims=True) + EPS
    tangent  = d1 / speed                                       # (N, 3)
    acc_par  = np.sum(acc * tangent, axis=1, keepdims=True) * tangent
    acc_perp = acc - acc_par

    preds = []
    for spec in CANDIDATES:
        t  = spec.time_scale
        t2 = t * t

        pred = (
            p0
            + spec.d1  * t  * d1
            + spec.d2  * t  * d2
            + spec.par  * t2 * acc_par
            + spec.perp * t2 * acc_perp
            + spec.jerk * t2 * jerk
        )
        preds.append(pred)

    return np.stack(preds, axis=1).astype(np.float32)   # (N, C, 3)


def make_seq_features(x: np.ndarray) -> np.ndarray:
    """
    Sequence features for each of 11 time steps.
    x: (N, 11, 3)
    returns: (N, 11, 11)  — 11 features: speed, prev_speed_ratio, acc_norm, acc_par, acc_perp,
                            jerk_norm, turn_cos, curvature, direction_flag, jerk_abs, acc_cos
    """
    N = x.shape[0]
    feats = []

    for t in range(1, 11):
        d1   = x[:, t]     - x[:, t - 1]
        speed = np.linalg.norm(d1, axis=1, keepdims=True) + EPS
        tangent = d1 / speed

        if t >= 2:
            d2   = x[:, t - 1] - x[:, t - 2]
            acc  = d1 - d2
            prev_speed = np.linalg.norm(d2, axis=1, keepdims=True) + EPS
            prev_speed_ratio = prev_speed / speed
            acc_par  = np.sum(acc * tangent, axis=1, keepdims=True)
            acc_perp_vec = acc - acc_par * tangent
            acc_perp = np.linalg.norm(acc_perp_vec, axis=1, keepdims=True)
            acc_norm = np.linalg.norm(acc, axis=1, keepdims=True)
            turn_cos = np.clip(np.sum(tangent * (d2 / (prev_speed)), axis=1, keepdims=True), -1, 1)
            curvature = acc_perp / (speed ** 2 + EPS)
        else:
            prev_speed_ratio = np.ones((N, 1), dtype=np.float32)
            acc_par  = np.zeros((N, 1), dtype=np.float32)
            acc_perp = np.zeros((N, 1), dtype=np.float32)
            acc_norm = np.zeros((N, 1), dtype=np.float32)
            turn_cos = np.ones((N, 1), dtype=np.float32)
            curvature = np.zeros((N, 1), dtype=np.float32)

        if t >= 3:
            d3             = x[:, t - 2] - x[:, t - 3]
            prev_acc       = d2 - d3
            jerk_norm      = np.linalg.norm(acc - prev_acc, axis=1, keepdims=True) / (speed + EPS)
            direction_flag = np.sign(acc_par)
            jerk_abs       = np.linalg.norm(acc - prev_acc, axis=1, keepdims=True) / 0.05
            prev_acc_norm  = np.linalg.norm(prev_acc, axis=1, keepdims=True) + EPS
            acc_cos        = np.sum(acc * prev_acc, axis=1, keepdims=True) / (acc_norm * prev_acc_norm + EPS)
        else:
            jerk_norm      = np.zeros((N, 1), dtype=np.float32)
            direction_flag = np.zeros((N, 1), dtype=np.float32)
            jerk_abs       = np.zeros((N, 1), dtype=np.float32)
            acc_cos        = np.zeros((N, 1), dtype=np.float32)

        step_feat = np.concatenate([
            speed / 0.05,                    # normalized speed
            prev_speed_ratio,
            acc_norm / (speed + EPS),
            acc_par  / (speed + EPS),
            acc_perp / (speed + EPS),
            jerk_norm,
            turn_cos,
            curvature * 10,
            direction_flag,
            jerk_abs,                        # absolute jerk size (jerk_xl 감지)
            acc_cos,                         # acc direction consistency (급격한 변화 감지)
        ], axis=1)  # (N, 11)
        feats.append(step_feat)

    # pad first step (t=0) with zeros
    feats = [np.zeros((N, 11), dtype=np.float32)] + feats  # 11 steps
    return np.stack(feats, axis=1).astype(np.float32)      # (N, 11, 11)


def make_cand_features(x: np.ndarray, cands: np.ndarray) -> np.ndarray:
    """
    Candidate-specific features.
    x:     (N, 11, 3)
    cands: (N, C, 3)
    returns: (N, C, 10)
    """
    p0, d1, d2, acc, jerk = motion_terms(x, end_idx=10)
    speed   = np.linalg.norm(d1, axis=1, keepdims=True) + EPS  # (N, 1)
    tangent = d1 / speed
    acc_par = np.sum(acc * tangent, axis=1, keepdims=True)
    acc_perp_vec = acc - acc_par * tangent
    acc_perp = np.linalg.norm(acc_perp_vec, axis=1, keepdims=True)

    horizon = 2.0
    delta   = cands - p0[:, np.newaxis, :]          # (N, C, 3)
    dist    = np.linalg.norm(delta, axis=2)          # (N, C)

    # project delta onto Frenet frame
    cand_par  = np.einsum('nci,ni->nc', delta, tangent)  # (N, C)
    cand_perp = dist - np.abs(cand_par)

    speed_h = speed[:, 0] * horizon  # (N,)

    feats = np.stack([
        cand_par  / (speed_h[:, np.newaxis] + EPS),
        cand_perp / (speed_h[:, np.newaxis] + EPS),
        dist      / (speed_h[:, np.newaxis] + EPS),
        np.array([s.d1  for s in CANDIDATES])[np.newaxis, :].repeat(len(x), axis=0),
        np.array([s.par for s in CANDIDATES])[np.newaxis, :].repeat(len(x), axis=0),
        np.array([s.perp for s in CANDIDATES])[np.newaxis, :].repeat(len(x), axis=0),
        np.array([s.d2  for s in CANDIDATES])[np.newaxis, :].repeat(len(x), axis=0),
        np.array([s.jerk for s in CANDIDATES])[np.newaxis, :].repeat(len(x), axis=0),
        np.array([s.time_scale for s in CANDIDATES])[np.newaxis, :].repeat(len(x), axis=0),
        (acc_par[:, 0, np.newaxis] / (speed[:, 0, np.newaxis] + EPS)).repeat(N_CANDIDATES, axis=1),
    ], axis=2).astype(np.float32)   # (N, C, 10)

    return feats


# ---------------------------------------------------------------------------
# GPU (torch) versions — same logic, batch operations on GPU
# ---------------------------------------------------------------------------

_CAND_PARAMS_CACHE: dict = {}

def _cand_params_gpu(device, dtype):
    key = (str(device), dtype)
    if key not in _CAND_PARAMS_CACHE:
        arr = np.array([[s.d1, s.d2, s.par, s.perp, s.jerk, s.time_scale]
                        for s in CANDIDATES], dtype=np.float32)
        _CAND_PARAMS_CACHE[key] = torch.tensor(arr, device=device, dtype=dtype)
    return _CAND_PARAMS_CACHE[key]  # (C, 6)


def make_candidates_gpu(x: torch.Tensor) -> torch.Tensor:
    """x: (B, 11, 3) → (B, C, 3)"""
    p0       = x[:, 10]
    d1       = x[:, 10] - x[:, 9]
    d2       = x[:, 9]  - x[:, 8]
    acc      = d1 - d2
    prev_acc = d2 - (x[:, 8] - x[:, 7])
    jerk     = acc - prev_acc

    speed    = d1.norm(dim=-1, keepdim=True).clamp(min=EPS)
    tangent  = d1 / speed
    acc_par  = (acc * tangent).sum(-1, keepdim=True) * tangent
    acc_perp = acc - acc_par

    p = _cand_params_gpu(x.device, x.dtype)             # (C, 6)
    d1_c, d2_c, par_c, pe_c, jk_c, ts_c = p.unbind(-1) # (C,) each
    ts2_c = ts_c ** 2

    return (
        p0[:, None]
        + (d1_c * ts_c)[None, :, None]   * d1[:, None]
        + (d2_c * ts_c)[None, :, None]   * d2[:, None]
        + (par_c * ts2_c)[None, :, None] * acc_par[:, None]
        + (pe_c  * ts2_c)[None, :, None] * acc_perp[:, None]
        + (jk_c  * ts2_c)[None, :, None] * jerk[:, None]
    )  # (B, C, 3)


def make_seq_features_gpu(x: torch.Tensor) -> torch.Tensor:
    """x: (B, 11, 3) → (B, 11, 11)"""
    B, dev, dt = x.shape[0], x.device, x.dtype
    feats = [torch.zeros(B, 11, device=dev, dtype=dt)]  # t=0 padding

    for t in range(1, 11):
        d1    = x[:, t] - x[:, t - 1]
        speed = d1.norm(dim=-1, keepdim=True).clamp(min=EPS)
        tang  = d1 / speed

        if t >= 2:
            d2               = x[:, t - 1] - x[:, t - 2]
            acc              = d1 - d2
            prev_speed       = d2.norm(dim=-1, keepdim=True).clamp(min=EPS)
            prev_speed_ratio = prev_speed / speed
            acc_par_s        = (acc * tang).sum(-1, keepdim=True)
            acc_par          = acc_par_s * tang
            acc_perp         = (acc - acc_par).norm(dim=-1, keepdim=True)
            acc_norm         = acc.norm(dim=-1, keepdim=True)
            turn_cos         = ((tang * (d2 / prev_speed)).sum(-1, keepdim=True)).clamp(-1, 1)
            curvature        = acc_perp / (speed ** 2 + EPS)
        else:
            prev_speed_ratio = torch.ones(B, 1, device=dev, dtype=dt)
            acc_par_s        = torch.zeros(B, 1, device=dev, dtype=dt)
            acc_perp         = torch.zeros(B, 1, device=dev, dtype=dt)
            acc_norm         = torch.zeros(B, 1, device=dev, dtype=dt)
            turn_cos         = torch.ones(B, 1, device=dev, dtype=dt)
            curvature        = torch.zeros(B, 1, device=dev, dtype=dt)

        if t >= 3:
            d3             = x[:, t - 2] - x[:, t - 3]
            prev_acc       = d2 - d3
            jerk_norm      = (acc - prev_acc).norm(dim=-1, keepdim=True) / (speed + EPS)
            direction_flag = torch.sign(acc_par_s)
            jerk_abs       = (acc - prev_acc).norm(dim=-1, keepdim=True) / 0.05
            prev_acc_norm  = prev_acc.norm(dim=-1, keepdim=True).clamp(min=EPS)
            acc_cos        = (acc * prev_acc).sum(-1, keepdim=True) / (acc_norm * prev_acc_norm + EPS)
        else:
            jerk_norm      = torch.zeros(B, 1, device=dev, dtype=dt)
            direction_flag = torch.zeros(B, 1, device=dev, dtype=dt)
            jerk_abs       = torch.zeros(B, 1, device=dev, dtype=dt)
            acc_cos        = torch.zeros(B, 1, device=dev, dtype=dt)

        feats.append(torch.cat([
            speed / 0.05,
            prev_speed_ratio,
            acc_norm  / (speed + EPS),
            acc_par_s / (speed + EPS),
            acc_perp  / (speed + EPS),
            jerk_norm,
            turn_cos,
            curvature * 10,
            direction_flag,
            jerk_abs,       # absolute jerk size (jerk_xl 감지)
            acc_cos,        # acc direction consistency (급격한 변화 감지)
        ], dim=-1))  # (B, 11)

    return torch.stack(feats, dim=1)  # (B, 11, 11)


def make_cand_features_gpu(x: torch.Tensor, cands: torch.Tensor) -> torch.Tensor:
    """x: (B,11,3), cands: (B,C,3) → (B,C,10)"""
    p0        = x[:, 10]
    d1        = x[:, 10] - x[:, 9]
    speed     = d1.norm(dim=-1, keepdim=True).clamp(min=EPS)
    tangent   = d1 / speed
    acc       = d1 - (x[:, 9] - x[:, 8])
    acc_par_s = (acc * tangent).sum(-1, keepdim=True)   # (B, 1)

    delta     = cands - p0[:, None]                     # (B, C, 3)
    dist      = delta.norm(dim=-1)                      # (B, C)
    cand_par  = torch.einsum('bci,bi->bc', delta, tangent)
    cand_perp = dist - cand_par.abs()
    speed_h   = speed[:, 0] * 2.0                       # (B,)

    p = _cand_params_gpu(x.device, x.dtype)             # (C, 6)
    d1_a, d2_a, par_a, pe_a, jk_a, ts_a = p.unbind(-1) # (C,) each

    return torch.stack([
        cand_par  / (speed_h[:, None] + EPS),
        cand_perp / (speed_h[:, None] + EPS),
        dist      / (speed_h[:, None] + EPS),
        d1_a [None].expand(len(x), -1),
        par_a[None].expand(len(x), -1),
        pe_a [None].expand(len(x), -1),
        d2_a [None].expand(len(x), -1),
        jk_a [None].expand(len(x), -1),
        ts_a [None].expand(len(x), -1),
        (acc_par_s / (speed + EPS)).expand(-1, N_CANDIDATES),
    ], dim=-1)  # (B, C, 10)
