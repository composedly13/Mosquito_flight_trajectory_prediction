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
EPOCHS      = 200
LR          = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE    = 40        # early stopping

# Augmentation: 'so3' | 'yaw' | 'yaw_speed' | 'none'
# 'yaw'       : z-axis rotation only (safe for LiDAR z=UP frame)
# 'yaw_speed' : yaw + speed-scale (scale displacements around last point p0)
# yaw_speed(0.85~1.15) 실험 결과: OOF -0.24pp (0.6489→0.6465), 효과 없음 → yaw 복귀
AUG_MODE          = 'yaw'
SPEED_SCALE_RANGE = (0.85, 1.15)   # Uniform scale range (실험 보존용)
SPEED_SCALE_PROB  = 0.5            # fraction of samples that get scaled (실험 보존용)

# Selector Model (Transformer)
# d256 run result: CV 0.6365 (worse than d128 0.6410) — likely overfitting on 8k samples
# reverting to d128 with yaw to isolate augmentation effect
D_MODEL     = 128
NHEAD       = 4
NUM_LAYERS  = 3
DROPOUT     = 0.1

# Boundary MLP — R-Hit@1cm 기준으로 실제 도움이 되는 범위만
# 0.9cm 미만: 이미 hit, 건드리면 miss로 바뀔 수 있음
# 1.3cm 초과: 6mm 보정으로 1cm 이하 달성 불가
BOUNDARY_LO = 0.009    # 0.9cm
BOUNDARY_HI = 0.013    # 1.3cm
CORRECTION_CAP = 0.006 # 최대 보정량 6mm

# Loss hyperparameters
# SOFT_TEMP history: 0.005(default) → 0.003(실험, -0.57pp) → 0.005(복귀)
SOFT_TEMP       = 0.005   # soft-label temperature
PAIRWISE_WEIGHT = 0.25    # pairwise ranking loss weight (0.5 시도 → fold5 붕괴, 유지)
# LISTMLE_WEIGHT grid: 0.0(baseline A) → 0.05(B) → 0.1(C) → 0.2(D)
LISTMLE_WEIGHT  = 0.05    # 원래 50-cand + LML=0.05 (multi-seed run: 42+123+777)
# Oracle Margin Loss weight
# focal(2.0) → oracle rank 악화 (13.3→16.4)
# oracle_margin(0.10) → OOF -0.85pp, B-group 42.6%→48.9%, rank 16.9 (soft-CE와 gradient 충돌)
# → 두 방식 모두 실패: loss 충돌 구조 확인
# 현재: 0.0 (CE+PW+LML 순수 config 복귀)
B_GROUP_WEIGHT  = 0.0

# Entropy Penalty weight — Phase 9 실험: oracle rank 17.6→24.7 역효과 확인 → 0.0 고정
ENTROPY_WEIGHT  = 0.0

# Family Classifier weight (Phase 10 2-stage 구조)
# Stage 1: CLS → 6 family 분류 (auxiliary CE loss)
# Stage 2: family-boosted logits → candidate ranking
FAMILY_WEIGHT   = 0.5

# Hit-aware BCE Loss
# 목적: oracle(1cm 이내) = 1, 나머지 = 0 이진 분류 → soft-CE보다 명확한 gradient
# pos_weight: oracle 후보(~2개) vs 비-oracle(~50개) 클래스 불균형 보정
# Phase 10 도입: soft-CE 대체 → selector가 직접 R-Hit@1cm 신호 학습
BCE_POS_WEIGHT  = 5.0

# Prediction
TOPK = 10   # train / predict / analyze 통일

# Metric
R_HIT_THRESHOLD = 0.01
EPS = 1e-8
