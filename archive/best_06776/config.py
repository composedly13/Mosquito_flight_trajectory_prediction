from pathlib import Path

# Paths — resolved relative to this file so scripts work from any cwd
_EXP_DIR        = Path(__file__).resolve().parent          # dev/experiments/
_ROOT           = _EXP_DIR.parent.parent                   # project root

DATA_DIR        = _ROOT / "data"
TRAIN_DIR       = DATA_DIR / "train"
TEST_DIR        = DATA_DIR / "test"
LABELS_PATH     = DATA_DIR / "train_labels.csv"
SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_DIR      = _EXP_DIR / "outputs"

# Training
SEED        = 42
N_FOLDS     = 5
BATCH_SIZE  = 256
EPOCHS      = 1000
LR          = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE    = 300        # Phase 5 재확립 기준 (seed42=40, seed777=80 — train.py --patience 80으로 오버라이드)

# Augmentation: 'so3' | 'yaw' | 'yaw_speed' | 'none'
# yaw-only: z=UP 보존, SO3보다 +4.8pp 확인
# yaw_speed(0.85~1.15): OOF -0.24pp → 효과 없음
AUG_MODE          = 'yaw'
SPEED_SCALE_RANGE = (0.85, 1.15)
SPEED_SCALE_PROB  = 0.5

# Phase 14 추가 증강 — Phase 5 회귀로 비활성화
# AUG_FLIP: seed123 안정성 미검증, Phase 5 baseline에 없음
# AUG_NOISE: 동일 이유
AUG_FLIP   = False
AUG_NOISE  = False
NOISE_STD  = 0.001  # 1mm (비활성 상태, 실험 보존용)

# Selector Model (Transformer)
# d256: CV 0.6365 (d128 0.6410보다 나쁨, 10k 샘플 과적합) → d128 유지
D_MODEL     = 128
NHEAD       = 4
NUM_LAYERS  = 3
DROPOUT     = 0.1

# Loss hyperparameters
SOFT_TEMP       = 0.005   # soft-label temperature (0.003 시도 → -0.57pp → 복귀)
PAIRWISE_WEIGHT = 0.25    # pairwise ranking loss (0.5 → fold5 붕괴 확인)
# LISTMLE grid: 0.0(A) → 0.05(B,+0.23pp) → 0.10(C,rank 10.0) → 0.20(D,불안정)
# Smart50-cand는 극단 후보 ranking 압력이 더 필요 → 0.10
LISTMLE_WEIGHT  = 0.10

# Prediction
TOPK = 10   # Top-10 weighted average (Top-7~10 최적 확인)

# Focal ListMLE (Step 3: B-group hard sample 강화)
# oracle 후보가 상위권에 없는 샘플에 더 큰 gradient 압력 부여
# FOCAL_LML_WEIGHT = 0.0: 비활성 (기존 LML×0.10만 사용)
# 권장 시작값: 0.05
# FOCAL_LML_MODE: "oracle_rank" | "margin"
FOCAL_LML_WEIGHT = 0.0     # 실험 시 0.05부터 시작
FOCAL_LML_GAMMA  = 1.0     # focal exponent (1.0 / 2.0)
FOCAL_LML_MODE   = "oracle_rank"   # "oracle_rank" | "margin"

# Candidate feature normalization (Step 2: cand_feat robust clipping)
# 목적: jerk_xxl/latency_s075/turn_p080 같은 극단 후보 feature scale 안정화
# → selector가 극단 후보를 이상치로 취급하는 문제 완화
# 'none': 기존 동작 (default)
# 'clip': cand_feat = clip(cand_feat, -V, +V)
# 'tanh': cand_feat = tanh(cand_feat / V)
CAND_FEAT_NORM_MODE  = "none"    # "none" | "clip" | "tanh"
CAND_FEAT_CLIP_VALUE = 2.5       # clip/tanh scale 값

# Experiment A: cand_feat physics interaction (Step A)
# "이 후보의 물리 가정이 실제 관측과 얼마나 맞는가"를 명시적 피처로 제공
# 추가 피처 4개: obs_acc_perp, par_match, perp_match, jerk_match → CAND_DIM 10→14
# False: 기존 CAND_DIM=10 (default, 저장된 모델 호환)
# True:  CAND_DIM=14 (실험 A 학습 시)
CAND_FEAT_INTERACTION = True

# Metric
R_HIT_THRESHOLD = 0.01
EPS = 1e-8
