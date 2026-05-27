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
PATIENCE    = 300       # 다음 실험부터 적용 (CosineAnnealingLR T_max=1000 충분한 LR decay)

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

# Metric
R_HIT_THRESHOLD = 0.01
EPS = 1e-8
