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
PATIENCE    = 30        # early stopping

# Selector Model (Transformer)
D_MODEL     = 128
NHEAD       = 4
NUM_LAYERS  = 3
DROPOUT     = 0.1

# Boundary MLP
BOUNDARY_LO = 0.005    # 0.5cm  이하: 이미 해결
BOUNDARY_HI = 0.025    # 2.5cm  이상: 복구 불가
CORRECTION_CAP = 0.006 # 최대 보정량 6mm

# Metric
R_HIT_THRESHOLD = 0.01
EPS = 1e-8
