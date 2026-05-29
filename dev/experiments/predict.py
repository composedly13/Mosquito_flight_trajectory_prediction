"""
Inference: K-Fold selector ensemble + optional multi-seed ensemble -> submission CSV.
Optional entropy-based regression blend or physics-routed LSTM blend.

Single seed:      python predict.py --seed 42
Multi-seed:       python predict.py --seeds 42 777
LSTM blend:       python predict.py --seed 42 --lstm
Reg2 blend:       python predict.py --seeds 42 777 --reg2 --beta 1.0
RegMLP blend:     python predict.py --seeds 42 777 --beta 1.0   (legacy)

LSTM blend (--lstm):
  alpha = physics_routing_alpha(seq_feat)  ∈ [0, 0.5]  per-sample
  pred  = (1 - alpha) × selector + alpha × lstm
  Routing fires on: decel (speed_ratio > 1.8) or jerk (jerk_abs > 0.8)
  → C-group 극감속/jerk 클러스터 타겟 (~50% of C-group = ~12.5% of total)

Entropy blend (--reg2):
  α = clip(1 - beta × H_norm, 0, 1)
  pred = α × selector + (1-α) × regressor
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
from regression import (
    load_reg_models, predict_reg_batch,
    load_reg2_models, predict_reg2_batch,
    entropy_blend,
    load_lstm_models, predict_lstm_batch, physics_routing_alpha,
)


def load_selectors(seeds: list, device: torch.device, out_tag: str = "") -> list:
    """Load all fold models for the given seeds.  seeds=[42] for single-seed inference."""
    models = []
    for seed in seeds:
        seed_dir = OUTPUT_DIR / f"seed{seed}{out_tag}"
        for fold in range(N_FOLDS):
            path = seed_dir / f"selector_fold{fold}.pt"
            if not path.exists():
                raise FileNotFoundError(f"Missing: {path}  (run: python train.py --seed {seed} --out-tag '{out_tag}')")
            m = CandidateSelector().to(device)
            m.load_state_dict(torch.load(path, map_location=device), strict=False)
            m.eval()
            models.append(m)
    print(f"  {len(models)} selector models loaded ({len(seeds)} seeds × {N_FOLDS} folds)")
    return models



def predict(
    seeds: list = None,
    beta: float = 1.0,
    temp: float = 2.0,
    use_reg2: bool = False,
    use_lstm: bool = False,
    out_tag: str = "",
    out_name: str = "",
):
    if seeds is None:
        seeds = [SEED]
    torch.manual_seed(seeds[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Seeds: {seeds}")

    selectors = load_selectors(seeds, device, out_tag=out_tag)

    # LSTM 모델 로드 (--lstm 플래그)
    lstm_models = []
    if use_lstm:
        print("  LSTM 모델 확인...")
        lstm_models = load_lstm_models(seeds, device)
        if not lstm_models:
            print("  ※ LSTM 없음 — selector-only로 진행 (--lstm 무시)")

    # Regressor loading: prefer reg2 (TransformerRegressor), fallback to RegMLP
    reg_models_flat = []
    reg2_label = ""
    if use_reg2:
        print("  TransformerRegressor(reg2) 모델 확인...")
        reg2_seed_models = load_reg2_models(seeds, device)
        reg_models_flat  = [m for fl in reg2_seed_models.values() for m in fl]
        reg2_label = "reg2"
    # RegMLP fallback 제거 — Phase 5에서 OOF 0.0681로 완전 실패 확인
    # blend는 --reg2 플래그 명시 시에만 활성화
    use_blend = len(reg_models_flat) > 0

    ids, coords, _ = load_all(TEST_DIR)
    N = len(ids)
    print(f"Test samples: {N}")

    all_preds   = []
    all_logits  = []
    all_reg     = [] if use_blend else None
    all_lstm    = [] if lstm_models else None
    all_alpha   = [] if lstm_models else None

    for start in tqdm(range(0, N, BATCH_SIZE), desc="Inference"):
        end  = min(start + BATCH_SIZE, N)
        c_np = coords[start:end]
        c    = torch.tensor(c_np).to(device)

        with torch.no_grad():
            cands_t    = make_candidates_gpu(c)
            seq_t      = make_seq_features_gpu(c)
            cand_t     = make_cand_features_gpu(c, cands_t)
            avg_logits = sum(m(seq_t, cand_t) for m in selectors) / len(selectors)
            # Stage 2-B: apply gate mask if enabled
            from config import C_GATE_V1_ENABLED
            if C_GATE_V1_ENABLED:
                from candidates import compute_gate_mask_gpu
                gate = compute_gate_mask_gpu(c)
                avg_logits = avg_logits.masked_fill(~gate, -1e9)
            pred       = selector_predict(avg_logits, cands_t, topk=TOPK, temp=temp)

        all_preds.append(pred.cpu().numpy())
        all_logits.append(avg_logits.cpu().numpy())

        if use_blend:
            if reg2_label == "reg2":
                reg_pred = predict_reg2_batch(reg_models_flat, c)
            else:
                reg_pred = predict_reg_batch(reg_models_flat, c)
            all_reg.append(reg_pred.cpu().numpy())

        if lstm_models:
            with torch.no_grad():
                lstm_pred = predict_lstm_batch(lstm_models, c, seq_t)
                alpha_t   = physics_routing_alpha(seq_t)    # seq_t 재사용
            all_lstm.append(lstm_pred.cpu().numpy())
            all_alpha.append(alpha_t.cpu().numpy())

    all_preds  = np.concatenate(all_preds,  axis=0)   # (N, 3)
    all_logits = np.concatenate(all_logits, axis=0)   # (N, C)
    if use_blend:
        all_reg = np.concatenate(all_reg, axis=0)      # (N, 3)
    if lstm_models:
        all_lstm  = np.concatenate(all_lstm,  axis=0)  # (N, 3)
        all_alpha = np.concatenate(all_alpha, axis=0)  # (N,)

    sub = pd.read_csv(SUBMISSION_PATH, index_col="id")

    def save_csv(preds: np.ndarray, name: str):
        df = pd.DataFrame(preds, index=ids, columns=sub.columns)
        df.index.name = "id"
        path = OUTPUT_DIR / name
        df.to_csv(path)
        print(f"  {path}  ({N} rows)")

    print("\n[제출 파일 생성]")

    # 1. Selector-only
    csv_name = out_name if out_name else "submission.csv"
    save_csv(all_preds, csv_name)

    # 2. Physics-routed LSTM blend
    if lstm_models:
        alpha    = all_alpha[:, np.newaxis]                            # (N, 1)
        n_routed = int((alpha > 0).sum())
        print(f"  Routing 발동: {n_routed} / {N} 샘플 ({n_routed/N*100:.1f}%)")
        print(f"  alpha 분포: mean={alpha.mean():.3f}  max={alpha.max():.3f}")

        lstm_blended = (1 - alpha) * all_preds + alpha * all_lstm
        save_csv(lstm_blended, "submission_lstm.csv")
        print(f"  ※ physics routing LSTM blend (decel/jerk → max α=0.5)")

    # 3. Entropy-blend (regressor 있을 때)
    if use_blend:
        blended = entropy_blend(all_preds, all_reg, all_logits, beta=beta)
        blend_name = f"submission_blend_{reg2_label}.csv"
        save_csv(blended, blend_name)
        print(f"  ※ entropy blend [{reg2_label}] β={beta}  (β 조정: --beta 값)")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--seed",  type=int, default=None,
                       help="Single seed (default: config.SEED)")
    group.add_argument("--seeds", type=int, nargs="+", default=None,
                       help="Multiple seeds, e.g. --seeds 42 777")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Entropy-blend strength: 0=pure selector (default: 1.0)")
    parser.add_argument("--temp", type=float, default=2.0,
                        help="Softmax temperature for Top-k weighted avg (default: 2.0)")
    parser.add_argument("--reg2", action="store_true",
                        help="Use TransformerRegressor (reg2) for entropy-blend instead of RegMLP")
    parser.add_argument("--lstm", action="store_true",
                        help="Physics-routed LSTM blend (train_lstm.py 학습 후 사용)")
    parser.add_argument("--out-tag", type=str, default="",
                        help="Suffix appended to model dir (e.g. '_lml05'). Must match --out-tag used in train.py.")
    parser.add_argument("--out-name", type=str, default="",
                        help="Output CSV filename (default: submission.csv). Use to avoid overwriting current best.")
    parser.add_argument("--gate-v1", action="store_true",
                        help="Enable Stage 2-B gated C-candidates v1 (56 cands)")
    parser.add_argument("--gate-jerk-thresh", type=float, default=None,
                        help="Override C_GATE_JERK_THRESH (default: config q95=1.038493)")
    parser.add_argument("--gate-turn-thresh", type=float, default=None,
                        help="Override C_GATE_TURN_THRESH (default: config q95=0.626477)")
    parser.add_argument("--v2-turn",    action="store_true", help="v2-A: turn_p110/n110")
    parser.add_argument("--v2-jerk",    action="store_true", help="v2-B: jerk_extreme±1.80")
    parser.add_argument("--v2-latency", action="store_true", help="v2-C: latency_s065/l130")
    parser.add_argument("--v3-jerk-turn",    action="store_true", help="v3-A: jerk+turn (60)")
    parser.add_argument("--v3-jerk-latency", action="store_true", help="v3-B: jerk+latency (60)")
    parser.add_argument("--v3-full",         action="store_true", help="v3-full: jerk+turn+lat (62)")
    parser.add_argument("--v2-jerk200",      action="store_true", help="jerk±2.00 (58)")
    parser.add_argument("--v2-jerk220",      action="store_true", help="jerk±2.20 (58)")
    parser.add_argument("--v2-turn-p",       action="store_true", help="turn_p110 only (59)")
    parser.add_argument("--v2-turn-n",       action="store_true", help="turn_n110 only (59)")
    parser.add_argument("--v2-lat-slow",     action="store_true", help="latency_s065 only (59)")
    parser.add_argument("--v2-lat-fast",     action="store_true", help="latency_l130 only (59)")
    args = parser.parse_args()

    # Stage 2-B gate threshold overrides
    if args.gate_jerk_thresh is not None or args.gate_turn_thresh is not None:
        import config as _cfg_th_p
        import candidates as _cands_th_p
        if args.gate_jerk_thresh is not None:
            _cfg_th_p.C_GATE_JERK_THRESH   = args.gate_jerk_thresh
            _cands_th_p.C_GATE_JERK_THRESH = args.gate_jerk_thresh
        if args.gate_turn_thresh is not None:
            _cfg_th_p.C_GATE_TURN_THRESH   = args.gate_turn_thresh
            _cands_th_p.C_GATE_TURN_THRESH = args.gate_turn_thresh

    # v3 / ablation shortcuts
    if args.v3_jerk_turn:    args.v2_jerk = True; args.v2_turn = True
    if args.v3_jerk_latency: args.v2_jerk = True; args.v2_latency = True
    if args.v3_full:         args.v2_jerk = True; args.v2_turn = True; args.v2_latency = True
    _v2f = ['v2_jerk','v2_turn','v2_latency','v2_jerk200','v2_jerk220',
            'v2_turn_p','v2_turn_n','v2_lat_slow','v2_lat_fast']
    _v2c = ['C_GATE_V2_JERK','C_GATE_V2_TURN','C_GATE_V2_LATENCY',
            'C_GATE_V2_JERK_200','C_GATE_V2_JERK_220',
            'C_GATE_V2_TURN_P','C_GATE_V2_TURN_N',
            'C_GATE_V2_LAT_SLOW','C_GATE_V2_LAT_FAST']
    if any(getattr(args, f, False) for f in _v2f):
        import config as _cfg_vp; import candidates as _cands_vp
        for _f, _c in zip(_v2f, _v2c):
            v = getattr(args, _f, False)
            setattr(_cfg_vp, _c, v); setattr(_cands_vp, _c, v)
        if not args.gate_v1: args.gate_v1 = True

    # Stage 2-B v2 flags
    if args.v2_turn or args.v2_jerk or args.v2_latency:
        import config as _cfg_v2p
        import candidates as _cands_v2p
        _cfg_v2p.C_GATE_V2_TURN    = args.v2_turn
        _cfg_v2p.C_GATE_V2_JERK    = args.v2_jerk
        _cfg_v2p.C_GATE_V2_LATENCY = args.v2_latency
        _cands_v2p.C_GATE_V2_TURN    = args.v2_turn
        _cands_v2p.C_GATE_V2_JERK    = args.v2_jerk
        _cands_v2p.C_GATE_V2_LATENCY = args.v2_latency
        if not args.gate_v1:
            args.gate_v1 = True

    # Stage 2-B gate patch (before loading models)
    if args.gate_v1:
        import config as _cfg_p
        import candidates as _cands_p
        _cfg_p.C_GATE_V1_ENABLED   = True
        _cands_p.C_GATE_V1_ENABLED = True
        from candidates import (CANDIDATES as _bc_p, _EXTRA_CANDIDATES_V1 as _ev1_p,
                                 _V2_JERK_EXTRA as _v2j_p, _V2_JERK_200_EXTRA as _v2j200_p,
                                 _V2_JERK_220_EXTRA as _v2j220_p,
                                 _V2_TURN_EXTRA as _v2t_p, _V2_TURN_P_EXTRA as _v2tp_p,
                                 _V2_TURN_N_EXTRA as _v2tn_p,
                                 _V2_LATENCY_EXTRA as _v2l_p,
                                 _V2_LAT_SLOW_EXTRA as _v2ls_p, _V2_LAT_FAST_EXTRA as _v2lf_p,
                                 N_CANDIDATES_BASE as _nb_p, _family_id as _fid_p)
        import numpy as _np_p
        if len(_cands_p.CANDIDATES) == _nb_p:
            _v2_ex_p = (
                (list(_v2j_p)    if _cands_p.C_GATE_V2_JERK     else []) +
                (list(_v2j200_p) if _cands_p.C_GATE_V2_JERK_200 else []) +
                (list(_v2j220_p) if _cands_p.C_GATE_V2_JERK_220 else []) +
                (list(_v2t_p)    if _cands_p.C_GATE_V2_TURN     else []) +
                (list(_v2tp_p)   if _cands_p.C_GATE_V2_TURN_P   else []) +
                (list(_v2tn_p)   if _cands_p.C_GATE_V2_TURN_N   else []) +
                (list(_v2l_p)    if _cands_p.C_GATE_V2_LATENCY  else []) +
                (list(_v2ls_p)   if _cands_p.C_GATE_V2_LAT_SLOW else []) +
                (list(_v2lf_p)   if _cands_p.C_GATE_V2_LAT_FAST else [])
            )
            _cands_p.CANDIDATES   = list(_bc_p) + list(_ev1_p) + list(_v2_ex_p)
            _cands_p.N_CANDIDATES = len(_cands_p.CANDIDATES)
            _cands_p.CANDIDATE_FAMILY = _np_p.array(
                [_fid_p(s.name) for s in _cands_p.CANDIDATES], dtype=_np_p.int64
            )
            _cands_p._CAND_PARAMS_CACHE.clear()

    if args.seeds:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [SEED]

    predict(seeds=seeds, beta=args.beta, temp=args.temp, use_reg2=args.reg2,
            use_lstm=args.lstm, out_tag=args.out_tag, out_name=args.out_name)
