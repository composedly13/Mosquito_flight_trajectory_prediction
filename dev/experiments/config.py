from pathlib import Path

# Paths
DATA_DIR     = Path("data")
TRAIN_DIR    = DATA_DIR / "train"
TEST_DIR     = DATA_DIR / "test"
LABELS_PATH  = DATA_DIR / "train_labels.csv"
SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
MODEL_SAVE   = Path("dev/experiments/best_transformer.pt")

# Training
SEED         = 42
VAL_RATIO    = 0.1
BATCH_SIZE   = 256
EPOCHS       = 300
LR           = 1e-3
WEIGHT_DECAY = 1e-4

# Model
D_MODEL      = 64
NHEAD        = 4
NUM_LAYERS   = 2
DROPOUT      = 0.1

# Physics baseline
BETA         = 0.6   # acceleration correction coefficient

# Metric
R_HIT_THRESHOLD = 0.01  # 1cm
