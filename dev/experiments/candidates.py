"""
Frenet-frame physics candidate generation.
Each candidate is a physically plausible prediction of position at +80ms.

Phase 5 회귀 (commit 66620d4):
  Smart 50-cand — 10개 제거 + 10개 교체, 총 50개 유지
  제거: acc_2d1_040/050, jerk_small±0.08, frenet_par090/100_p000 + par085/095_n005,
        latency_s088/l112
  추가: jerk_xxl±0.80, latency_s075/s080/l120, turn_p070/n070/p080/n080,
        frenet_par130_n020
  Oracle: 77.02% (vs original-50 74.89%)
  Phase 5 2-seed(42+777) LB: 0.6716 (전체 best)
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as _F
from config import (EPS, CAND_FEAT_NORM_MODE, CAND_FEAT_CLIP_VALUE,
                    CAND_FEAT_INTERACTION, CAND_FEAT_SIGN,
                    C_GATE_V1_ENABLED, C_GATE_JERK_THRESH, C_GATE_TURN_THRESH,
                    C_GATE_V2_TURN, C_GATE_V2_JERK, C_GATE_V2_LATENCY,
                    C_GATE_V2_JERK_200, C_GATE_V2_JERK_220,
                    C_GATE_V2_TURN_P, C_GATE_V2_TURN_N,
                    C_GATE_V2_LAT_SLOW, C_GATE_V2_LAT_FAST)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    d1: float           # velocity scale (2.0 = pure linear extrapolation)
    par: float          # acceleration parallel component scale
    perp: float         # acceleration perpendicular component scale
    d2: float = 0.0     # previous velocity scale
    jerk: float = 0.0   # jerk scale
    time_scale: float = 1.0  # latency correction


CANDIDATES = [
    # Base
    CandidateSpec("p0_2d1",            2.00,  0.00,  0.00),

    # Acceleration family (강한 것만 유지, acc_040/050 제거)
    CandidateSpec("acc_2d1_056",       1.98,  0.56,  0.56),
    CandidateSpec("acc_2d1_060",       2.00,  0.60,  0.60),

    # Frenet family (핵심 — 중복 4개 제거: par090_p000, par100_p000, par085_n005, par095_n005)
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
    CandidateSpec("frenet_par130_n020",1.92,  1.30, -0.20),  # C-group par>1.20 20.4%

    # Jerk family (small±0.08 제거, xxl±0.80 추가)
    CandidateSpec("jerk_med_pos",      1.98,  0.90, -0.05, jerk= 0.15),
    CandidateSpec("jerk_med_neg",      1.98,  0.90, -0.05, jerk=-0.15),
    CandidateSpec("jerk_l_pos",        1.98,  0.88,  0.00, jerk= 0.30),
    CandidateSpec("jerk_l_neg",        1.98,  0.88,  0.00, jerk=-0.30),
    CandidateSpec("jerk_xl_pos",       1.96,  0.82,  0.00, jerk= 0.50),
    CandidateSpec("jerk_xl_neg",       1.96,  0.82,  0.00, jerk=-0.50),
    CandidateSpec("jerk_xxl_pos",      1.94,  0.76,  0.00, jerk= 0.80),  # C-group jerk p75=0.55
    CandidateSpec("jerk_xxl_neg",      1.94,  0.76,  0.00, jerk=-0.80),

    # Latency family (s088/l112 제거, s075/s080/l120 추가)
    CandidateSpec("latency_s075",      1.98,  0.96, -0.08, time_scale=0.75),
    CandidateSpec("latency_s080",      1.98,  0.96, -0.08, time_scale=0.80),  # C-group nearest 21.8%
    CandidateSpec("latency_s085",      1.98,  0.96, -0.08, time_scale=0.85),
    CandidateSpec("latency_s092",      1.98,  0.96, -0.08, time_scale=0.92),
    CandidateSpec("latency_l108",      1.98,  0.96, -0.08, time_scale=1.08),
    CandidateSpec("latency_l115",      1.98,  0.96, -0.08, time_scale=1.15),
    CandidateSpec("latency_l120",      1.98,  0.96, -0.08, time_scale=1.20),
    CandidateSpec("latency_l110_turn", 1.98,  1.10, -0.20, time_scale=1.10),
    CandidateSpec("latency_s090_turn", 1.96,  0.90,  0.20, time_scale=0.90),

    # 추가 Frenet
    CandidateSpec("frenet_par105_n010",2.00,  1.05, -0.10),
    CandidateSpec("frenet_par075_p010",1.98,  0.75,  0.10),

    # 급격한 방향 전환 (p070/n070/p080/n080 추가)
    CandidateSpec("turn_p030",         2.00,  0.80,  0.30),
    CandidateSpec("turn_n030",         2.00,  0.80, -0.30),
    CandidateSpec("turn_p045",         1.96,  0.70,  0.45),
    CandidateSpec("turn_n045",         1.96,  0.70, -0.45),
    CandidateSpec("turn_p060",         1.90,  0.55,  0.60),
    CandidateSpec("turn_n060",         1.90,  0.55, -0.60),
    CandidateSpec("turn_p070",         1.84,  0.45,  0.70),  # C-group |perp|>0.60 15.5%
    CandidateSpec("turn_n070",         1.84,  0.45, -0.70),
    CandidateSpec("turn_p080",         1.78,  0.38,  0.80),  # C-group |perp| p95=1.02
    CandidateSpec("turn_n080",         1.78,  0.38, -0.80),

    # Turn + Jerk 복합
    CandidateSpec("tj_pp20",           1.98,  0.85,  0.20, jerk= 0.20),
    CandidateSpec("tj_np20",           1.98,  0.85, -0.20, jerk= 0.20),
    CandidateSpec("tj_pn20",           1.98,  0.85,  0.20, jerk=-0.20),
    CandidateSpec("tj_nn20",           1.98,  0.85, -0.20, jerk=-0.20),
    CandidateSpec("turn_fast_n030",    2.08,  0.80, -0.30),
]

N_CANDIDATES_BASE = len(CANDIDATES)  # 50 — always Smart50

# ── Stage 2-B: Extra C-candidates (gated by physics) ─────────────────────────
# 활성화: config.C_GATE_V1_ENABLED = True
# turn_p090/n090  : obs_acc_perp >= C_GATE_TURN_THRESH (q95)
# jerk_extreme_*  : obs_jerk_abs  >= C_GATE_JERK_THRESH (q95)
# latency_s070/l125: JT (either condition above)
_EXTRA_CANDIDATES_V1 = [
    CandidateSpec("turn_p090",        1.72,  0.30,  0.90),            # sharp left turn
    CandidateSpec("turn_n090",        1.72,  0.30, -0.90),            # sharp right turn
    CandidateSpec("jerk_extreme_pos", 1.90,  0.70,  0.00, jerk= 1.20),# extreme positive jerk
    CandidateSpec("jerk_extreme_neg", 1.90,  0.70,  0.00, jerk=-1.20),# extreme negative jerk
    CandidateSpec("latency_s070",     1.98,  0.96, -0.08, time_scale=0.70),  # very slow timing
    CandidateSpec("latency_l125",     1.98,  0.96, -0.08, time_scale=1.25),  # very fast timing
]
N_EXTRA_V1 = len(_EXTRA_CANDIDATES_V1)

# ── Stage 2-B v2: Additional gated extra candidates ───────────────────────────
# Each set adds 2 candidates on top of v1 extra 6 (total 58 candidates when active)
# Gate: same JT_q95 threshold as v1 (obs_jerk >= 1.038493 OR obs_acc_perp >= 0.626477)
#
# Candidate param extrapolation from v1 extremes:
#   turn_p090 → turn_p110: each +10pp perp: d1 -0.06, par -0.08
#   jerk_extreme(1.20) → 1.80: each +0.40 jerk: d1 -0.03, par -0.045
#   latency: same d1/par/perp, just extend time_scale
_V2_TURN_EXTRA = [
    CandidateSpec("turn_p110",            1.60,  0.14,  1.10),   # sharp left turn v2
    CandidateSpec("turn_n110",            1.60,  0.14, -1.10),   # sharp right turn v2
]
_V2_JERK_EXTRA = [
    CandidateSpec("jerk_extreme_pos180",  1.84,  0.61,  0.00, jerk= 1.80),  # extreme positive jerk v2
    CandidateSpec("jerk_extreme_neg180",  1.84,  0.61,  0.00, jerk=-1.80),  # extreme negative jerk v2
]
_V2_LATENCY_EXTRA = [
    CandidateSpec("latency_s065",         1.98,  0.96, -0.08, time_scale=0.65),  # very slow timing v2
    CandidateSpec("latency_l130",         1.98,  0.96, -0.08, time_scale=1.30),  # very fast timing v2
]
N_EXTRA_V2 = 2  # each v2 variant adds 2 candidates

# ── Jerk strength ablation (±2.00, ±2.20) ────────────────────────────────────
# jerk±1.20(v1) → ±1.80(v2) → ±2.00 → ±2.20(last attack)
# param extrapolation: each +0.40: d1 -0.03, par -0.045
_V2_JERK_200_EXTRA = [
    CandidateSpec("jerk_extreme_pos200",  1.80,  0.565, 0.00, jerk= 2.00),
    CandidateSpec("jerk_extreme_neg200",  1.80,  0.565, 0.00, jerk=-2.00),
]
_V2_JERK_220_EXTRA = [
    CandidateSpec("jerk_extreme_pos220",  1.77,  0.52,  0.00, jerk= 2.20),
    CandidateSpec("jerk_extreme_neg220",  1.77,  0.52,  0.00, jerk=-2.20),
]

# ── One-sided turn/latency (v2_jerk base + 1 candidate) ──────────────────────
_V2_TURN_P_EXTRA  = [CandidateSpec("turn_p110", 1.60, 0.14,  1.10)]
_V2_TURN_N_EXTRA  = [CandidateSpec("turn_n110", 1.60, 0.14, -1.10)]
_V2_LAT_SLOW_EXTRA = [CandidateSpec("latency_s065", 1.98, 0.96, -0.08, time_scale=0.65)]
_V2_LAT_FAST_EXTRA = [CandidateSpec("latency_l130", 1.98, 0.96, -0.08, time_scale=1.30)]

# Effective candidate list — extended when gate enabled
# v2 variants build on v1 (50 base + 6 v1 + 2 v2 = 58)
_active_v2 = (
    _V2_TURN_EXTRA    if C_GATE_V2_TURN    else
    _V2_JERK_EXTRA    if C_GATE_V2_JERK    else
    _V2_LATENCY_EXTRA if C_GATE_V2_LATENCY else
    []
)
CANDIDATES    = CANDIDATES + (_EXTRA_CANDIDATES_V1 if C_GATE_V1_ENABLED else []) + _active_v2
N_CANDIDATES  = len(CANDIDATES)

FAMILY_NAMES = ["base", "acc", "frenet", "turn", "jerk", "latency"]

def _family_id(name: str) -> int:
    if name == "p0_2d1":              return 0
    if name.startswith("acc_"):       return 1
    if name.startswith("latency"):    return 5
    if "jerk" in name:                return 4
    if any(x in name for x in ["fast", "slow", "par070", "par120", "par110",
                                 "par075", "par105", "par130", "turn_", "tj_"]):
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
    """x: (N, 11, 3) → (N, C, 3)"""
    p0, d1, d2, acc, jerk = motion_terms(x, end_idx=10)

    speed    = np.linalg.norm(d1, axis=1, keepdims=True) + EPS
    tangent  = d1 / speed
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
    """x: (N, 11, 3) → (N, 11, 11)"""
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
            speed / 0.05,
            prev_speed_ratio,
            acc_norm / (speed + EPS),
            acc_par  / (speed + EPS),
            acc_perp / (speed + EPS),
            jerk_norm,
            turn_cos,
            curvature * 10,
            direction_flag,
            jerk_abs,
            acc_cos,
        ], axis=1)  # (N, 11)
        feats.append(step_feat)

    feats = [np.zeros((N, 11), dtype=np.float32)] + feats
    return np.stack(feats, axis=1).astype(np.float32)      # (N, 11, 11)


def make_cand_features(x: np.ndarray, cands: np.ndarray) -> np.ndarray:
    """x: (N,11,3), cands: (N,C,3) → (N,C,10) or (N,C,14) if CAND_FEAT_INTERACTION"""
    p0, d1, d2, acc, jerk = motion_terms(x, end_idx=10)
    speed   = np.linalg.norm(d1, axis=1, keepdims=True) + EPS
    tangent = d1 / speed
    acc_par = np.sum(acc * tangent, axis=1, keepdims=True)

    horizon = 2.0
    delta   = cands - p0[:, np.newaxis, :]
    dist    = np.linalg.norm(delta, axis=2)
    cand_par  = np.einsum('nci,ni->nc', delta, tangent)
    cand_perp = dist - np.abs(cand_par)
    speed_h = speed[:, 0] * horizon

    par_a  = np.array([s.par        for s in CANDIDATES])
    pe_a   = np.array([s.perp       for s in CANDIDATES])
    jk_a   = np.array([s.jerk       for s in CANDIDATES])

    feats = np.stack([
        cand_par  / (speed_h[:, np.newaxis] + EPS),
        cand_perp / (speed_h[:, np.newaxis] + EPS),
        dist      / (speed_h[:, np.newaxis] + EPS),
        np.array([s.d1        for s in CANDIDATES])[np.newaxis, :].repeat(len(x), axis=0),
        par_a[np.newaxis, :].repeat(len(x), axis=0),
        pe_a [np.newaxis, :].repeat(len(x), axis=0),
        np.array([s.d2        for s in CANDIDATES])[np.newaxis, :].repeat(len(x), axis=0),
        jk_a [np.newaxis, :].repeat(len(x), axis=0),
        np.array([s.time_scale for s in CANDIDATES])[np.newaxis, :].repeat(len(x), axis=0),
        (acc_par[:, 0, np.newaxis] / (speed[:, 0, np.newaxis] + EPS)).repeat(N_CANDIDATES, axis=1),
    ], axis=2).astype(np.float32)  # (N, C, 10)

    if CAND_FEAT_INTERACTION:
        # obs_acc_perp: 법선 가속도 크기 (정규화)
        acc_perp_v = acc - acc_par * tangent
        acc_perp_s = np.linalg.norm(acc_perp_v, axis=1, keepdims=True) / (speed + EPS)  # (N,1)
        # obs_jerk_par: 탄젠트 방향 jerk (정규화)
        obs_jerk_par = np.sum(jerk * tangent, axis=1, keepdims=True) / (speed + EPS)   # (N,1)
        # Normalized values expanded to all candidates
        obs_acc_par_n  = (acc_par[:, 0, np.newaxis] / (speed[:, 0, np.newaxis] + EPS)).repeat(N_CANDIDATES, axis=1)
        obs_acc_perp_n = acc_perp_s[:, 0, np.newaxis].repeat(N_CANDIDATES, axis=1)
        obs_jerk_n     = obs_jerk_par[:, 0, np.newaxis].repeat(N_CANDIDATES, axis=1)
        # Interaction features
        par_match  = par_a[np.newaxis, :] * obs_acc_par_n   # (N, C)
        perp_match = pe_a [np.newaxis, :] * obs_acc_perp_n  # (N, C)
        jerk_match = jk_a [np.newaxis, :] * obs_jerk_n      # (N, C)

        interact = np.stack([obs_acc_perp_n, par_match, perp_match, jerk_match],
                            axis=2).astype(np.float32)  # (N, C, 4)
        feats = np.concatenate([feats, interact], axis=2)   # (N, C, 14)

    if CAND_FEAT_SIGN:
        # Sign-aware features (Stage 2-A): CAND_DIM 14→16
        # Requires CAND_FEAT_INTERACTION=True (acc_perp_v, obs_jerk_par already computed)
        # 1. jerk_sign_match = sign(cand_jerk) × sign(obs_jerk_par) ∈ {-1,0,+1}
        obs_jk_par_1d = np.sum(jerk * tangent, axis=1) / (speed[:, 0] + EPS)  # (N,)
        jk_sign_mat   = np.sign(jk_a)[np.newaxis, :] * np.sign(obs_jk_par_1d)[:, np.newaxis]  # (N,C)

        # 2. perp_sign_match = sign(cand_perp) × sign(obs_jerk_perp projected onto acc_perp dir)
        #    obs_jk_perp_signed: tells which way the turn is about to change
        if not CAND_FEAT_INTERACTION:
            acc_perp_v = acc - acc_par * tangent  # fallback if not already computed
        acc_perp_mag = np.linalg.norm(acc_perp_v, axis=1, keepdims=True).clip(EPS)
        acc_perp_unit = acc_perp_v / acc_perp_mag                               # (N,3)
        jk_par_s      = np.sum(jerk * tangent, axis=1, keepdims=True)           # (N,1) scalar
        jk_perp_v     = jerk - jk_par_s * tangent                               # (N,3)
        obs_jk_perp_signed = np.sum(jk_perp_v * acc_perp_unit, axis=1)         # (N,) signed
        pe_sign_mat   = np.sign(pe_a)[np.newaxis, :] * np.sign(obs_jk_perp_signed)[:, np.newaxis]  # (N,C)

        sign_feats = np.stack([jk_sign_mat, pe_sign_mat], axis=2).astype(np.float32)  # (N,C,2)
        feats = np.concatenate([feats, sign_feats], axis=2)  # (N, C, 16)

    return feats


# ---------------------------------------------------------------------------
# GPU (torch) versions
# ---------------------------------------------------------------------------

_CAND_PARAMS_CACHE: dict = {}

def _cand_params_gpu(device, dtype):
    """(C, 6): d1, d2, par, perp, jerk, time_scale"""
    key = (str(device), dtype)
    if key not in _CAND_PARAMS_CACHE:
        arr = np.array(
            [[s.d1, s.d2, s.par, s.perp, s.jerk, s.time_scale]
             for s in CANDIDATES],
            dtype=np.float32,
        )
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

    p = _cand_params_gpu(x.device, x.dtype)               # (C, 6)
    d1_c, d2_c, par_c, pe_c, jk_c, ts_c = p.unbind(-1)   # (C,) each
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
    feats = [torch.zeros(B, 11, device=dev, dtype=dt)]

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
            jerk_abs,
            acc_cos,
        ], dim=-1))

    return torch.stack(feats, dim=1)  # (B, 11, 11)


def make_cand_features_gpu(x: torch.Tensor, cands: torch.Tensor) -> torch.Tensor:
    """x: (B,11,3), cands: (B,C,3) → (B,C,10) or (B,C,14) if CAND_FEAT_INTERACTION"""
    p0        = x[:, 10]
    d1        = x[:, 10] - x[:, 9]
    d2        = x[:, 9]  - x[:, 8]
    speed     = d1.norm(dim=-1, keepdim=True).clamp(min=EPS)
    tangent   = d1 / speed
    acc       = d1 - d2
    acc_par_s = (acc * tangent).sum(-1, keepdim=True)

    delta     = cands - p0[:, None]
    dist      = delta.norm(dim=-1)
    cand_par  = torch.einsum('bci,bi->bc', delta, tangent)
    cand_perp = dist - cand_par.abs()
    speed_h   = speed[:, 0] * 2.0

    p = _cand_params_gpu(x.device, x.dtype)               # (C, 6)
    d1_a, d2_a, par_a, pe_a, jk_a, ts_a = p.unbind(-1)

    feat = torch.stack([
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

    if CAND_FEAT_INTERACTION:
        # obs_acc_perp: 법선 가속도 크기 (정규화)
        acc_perp_s = (acc - acc_par_s * tangent).norm(dim=-1, keepdim=True) / (speed + EPS)  # (B, 1)
        # obs_jerk_par: 탄젠트 방향 jerk 부호값 (정규화)
        d3        = x[:, 8] - x[:, 7]
        prev_acc  = d2 - d3
        jerk_obs  = acc - prev_acc
        obs_jerk_par = (jerk_obs * tangent).sum(-1, keepdim=True) / (speed + EPS)  # (B, 1)
        # Normalized observed values → expanded to all candidates
        obs_acc_par_n  = (acc_par_s / (speed + EPS)).expand(-1, N_CANDIDATES)  # (B, C)
        obs_acc_perp_n = acc_perp_s.expand(-1, N_CANDIDATES)                   # (B, C)
        obs_jerk_n     = obs_jerk_par.expand(-1, N_CANDIDATES)                 # (B, C)
        # Interaction: candidate param × observed motion stat → per-candidate signal
        par_match  = par_a[None].expand(len(x), -1) * obs_acc_par_n   # (B, C)
        perp_match = pe_a [None].expand(len(x), -1) * obs_acc_perp_n  # (B, C)
        jerk_match = jk_a [None].expand(len(x), -1) * obs_jerk_n      # (B, C)

        interact = torch.stack(
            [obs_acc_perp_n, par_match, perp_match, jerk_match], dim=-1
        )  # (B, C, 4)
        feat = torch.cat([feat, interact], dim=-1)  # (B, C, 14)

    if CAND_FEAT_SIGN:
        # Sign-aware features (Stage 2-A): CAND_DIM 14→16
        # Requires CAND_FEAT_INTERACTION (jerk_obs, acc_perp_s already computed above)
        # 1. jerk_sign_match = sign(cand_jerk) × sign(obs_jerk_par) ∈ {-1,0,+1}
        obs_jk_par_1d = (jerk_obs * tangent).sum(-1) / (speed[:, 0] + EPS)  # (B,) signed
        jk_sign_c   = torch.sign(jk_a[None].expand(len(x), -1))             # (B, C)
        obs_jk_sign = torch.sign(obs_jk_par_1d[:, None].expand(-1, N_CANDIDATES))  # (B, C)
        jk_sign_mat = jk_sign_c * obs_jk_sign                               # (B, C)

        # 2. perp_sign_match = sign(cand_perp) × sign(obs_jerk_perp projected onto acc_perp dir)
        acc_perp_v  = acc - acc_par_s * tangent                             # (B, 3)
        acc_perp_m  = acc_perp_v.norm(dim=-1, keepdim=True).clamp(EPS)
        acc_perp_unit = acc_perp_v / acc_perp_m                            # (B, 3)
        jk_par_s_v  = (jerk_obs * tangent).sum(-1, keepdim=True)           # (B, 1)
        jk_perp_v   = jerk_obs - jk_par_s_v * tangent                     # (B, 3)
        obs_jk_perp = (jk_perp_v * acc_perp_unit).sum(-1)                 # (B,) signed
        pe_sign_c   = torch.sign(pe_a[None].expand(len(x), -1))           # (B, C)
        obs_pe_sign = torch.sign(obs_jk_perp[:, None].expand(-1, N_CANDIDATES))  # (B, C)
        pe_sign_mat = pe_sign_c * obs_pe_sign                              # (B, C)

        sign_feat = torch.stack([jk_sign_mat, pe_sign_mat], dim=-1)        # (B, C, 2)
        feat = torch.cat([feat, sign_feat], dim=-1)                        # (B, C, 16)

    # Robust normalization (config.CAND_FEAT_NORM_MODE)
    if CAND_FEAT_NORM_MODE == "clip":
        feat = feat.clamp(-CAND_FEAT_CLIP_VALUE, CAND_FEAT_CLIP_VALUE)
    elif CAND_FEAT_NORM_MODE == "tanh":
        feat = torch.tanh(feat / CAND_FEAT_CLIP_VALUE)
    # "none": no-op

    return feat


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2-B: Gate mask  (used when C_GATE_V1_ENABLED=True)
# ─────────────────────────────────────────────────────────────────────────────

def _gate_physics_np(x: np.ndarray):
    """Returns (obs_jerk, obs_perp) normalized by speed, shape (N,)."""
    p0_, d1_, d2_, acc_, jk_v = motion_terms(x, end_idx=10)
    sp_   = np.linalg.norm(d1_, axis=1, keepdims=True).clip(EPS)
    tg_   = d1_ / sp_
    aps_  = (acc_ * tg_).sum(axis=1, keepdims=True)
    apv_  = acc_ - aps_ * tg_
    obs_jerk = np.linalg.norm(jk_v,  axis=1) / (sp_[:, 0] + EPS)
    obs_perp = np.linalg.norm(apv_,  axis=1) / (sp_[:, 0] + EPS)
    return obs_jerk, obs_perp


def compute_gate_mask_np(x: np.ndarray,
                          jerk_thresh: float | None = None,
                          turn_thresh: float | None = None) -> np.ndarray:
    """
    x: (N, 11, 3)
    Returns: (N, N_CANDIDATES_BASE + N_EXTRA_V1) bool
    Base 50 candidates: always True.
    Extra 6 candidates: activated by physics gate.
    Thresholds read dynamically from module (supports CLI override).
    """
    # Read dynamically — NOT as default param (frozen at import time)
    import candidates as _self
    if jerk_thresh is None: jerk_thresh = _self.C_GATE_JERK_THRESH
    if turn_thresh is None:  turn_thresh = _self.C_GATE_TURN_THRESH

    N = len(x)
    n_b = N_CANDIDATES_BASE
    mask = np.ones((N, n_b + N_EXTRA_V1), dtype=bool)

    obs_jerk, obs_perp = _gate_physics_np(x)
    turn_gate  = obs_perp >= turn_thresh
    jerk_gate  = obs_jerk >= jerk_thresh
    jt_gate    = turn_gate | jerk_gate

    # v1 extra (indices n_b+0 … n_b+5)
    mask[:, n_b:n_b+2]   = turn_gate[:, None]   # turn_p090, turn_n090
    mask[:, n_b+2:n_b+4] = jerk_gate[:, None]   # jerk_extreme_pos/neg
    mask[:, n_b+4:n_b+6] = jt_gate[:, None]     # latency_s070, l125

    # v2 extra — additive slots after v1 (n_b+6 onward)
    import candidates as _s
    _slot = n_b + 6
    for _flag, _gate, _n in [
        (_s.C_GATE_V2_JERK,      jerk_gate, 2),
        (_s.C_GATE_V2_JERK_200,  jerk_gate, 2),
        (_s.C_GATE_V2_JERK_220,  jerk_gate, 2),
        (_s.C_GATE_V2_TURN,      turn_gate, 2),
        (_s.C_GATE_V2_TURN_P,    turn_gate, 1),
        (_s.C_GATE_V2_TURN_N,    turn_gate, 1),
        (_s.C_GATE_V2_LATENCY,   jt_gate,   2),
        (_s.C_GATE_V2_LAT_SLOW,  jt_gate,   1),
        (_s.C_GATE_V2_LAT_FAST,  jt_gate,   1),
    ]:
        if _flag:
            mask[:, _slot:_slot+_n] = _gate[:, None]
            _slot += _n
    return mask


def compute_gate_mask_gpu(x: torch.Tensor,
                           jerk_thresh: float | None = None,
                           turn_thresh: float | None = None) -> torch.Tensor:
    """
    x: (B, 11, 3)
    Returns: (B, N_CANDIDATES_BASE + N_EXTRA_V1) bool
    """
    # Read dynamically — NOT as default param (frozen at import time)
    import candidates as _self
    if jerk_thresh is None: jerk_thresh = _self.C_GATE_JERK_THRESH
    if turn_thresh is None:  turn_thresh = _self.C_GATE_TURN_THRESH

    B   = x.shape[0]
    n_b = N_CANDIDATES_BASE
    dev = x.device

    p0_ = x[:, 10]; d1_ = x[:, 10] - x[:, 9]; d2_ = x[:, 9] - x[:, 8]
    acc_    = d1_ - d2_
    d3_     = x[:, 8] - x[:, 7]
    prev_ac = d2_ - d3_
    jk_v    = acc_ - prev_ac                                              # (B, 3)
    sp_     = d1_.norm(dim=-1, keepdim=True).clamp(EPS)
    tg_     = d1_ / sp_
    aps_    = (acc_ * tg_).sum(-1, keepdim=True)
    apv_    = acc_ - aps_ * tg_

    obs_jerk = jk_v.norm(dim=-1) / (sp_[:, 0] + EPS)   # (B,)
    obs_perp = apv_.norm(dim=-1) / (sp_[:, 0] + EPS)   # (B,)

    turn_gate = obs_perp >= turn_thresh   # (B,)
    jerk_gate = obs_jerk >= jerk_thresh
    jt_gate   = turn_gate | jerk_gate

    import candidates as _sg
    _slot = n_b + 6
    # Count total v2 slots
    _v2_info = [
        (_sg.C_GATE_V2_JERK,     jerk_gate, 2),
        (_sg.C_GATE_V2_JERK_200, jerk_gate, 2),
        (_sg.C_GATE_V2_JERK_220, jerk_gate, 2),
        (_sg.C_GATE_V2_TURN,     turn_gate, 2),
        (_sg.C_GATE_V2_TURN_P,   turn_gate, 1),
        (_sg.C_GATE_V2_TURN_N,   turn_gate, 1),
        (_sg.C_GATE_V2_LATENCY,  jt_gate,   2),
        (_sg.C_GATE_V2_LAT_SLOW, jt_gate,   1),
        (_sg.C_GATE_V2_LAT_FAST, jt_gate,   1),
    ]
    n_v2 = sum(n for f, _, n in _v2_info if f)
    mask = torch.ones(B, n_b + N_EXTRA_V1 + n_v2, dtype=torch.bool, device=dev)
    mask[:, n_b:n_b+2]   = turn_gate.unsqueeze(1)
    mask[:, n_b+2:n_b+4] = jerk_gate.unsqueeze(1)
    mask[:, n_b+4:n_b+6] = jt_gate.unsqueeze(1)
    for _f, _g, _n in _v2_info:
        if _f:
            mask[:, _slot:_slot+_n] = _g.unsqueeze(1)
            _slot += _n
    return mask
