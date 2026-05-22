from pathlib import Path

# Paths
DATA_DIR        = Path("data")
TRAIN_DIR       = DATA_DIR / "train"
TEST_DIR        = DATA_DIR / "test"
LABELS_PATH     = DATA_DIR / "train_labels.csv"
SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_DIR      = Path("dev/experiments/outputs")

# Training
SEED        = 42
N_FOLDS     = 5
BATCH_SIZE  = 256
EPOCHS      = 200
LR          = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE    = 40        # early stopping

# Augmentation: 'so3' (random 3D rotation) | 'yaw' (z-axis rotation only) | 'none'
# 'yaw' is safer when z=UP has physical meaning (LiDAR coordinate frame)
AUG_MODE    = 'yaw'

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
SOFT_TEMP       = 0.005   # soft-label temperature (smaller = sharper targets)
PAIRWISE_WEIGHT = 0.25    # pairwise ranking loss weight

# Metric
R_HIT_THRESHOLD = 0.01
EPS = 1e-8
