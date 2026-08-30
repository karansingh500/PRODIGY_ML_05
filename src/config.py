from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATASET_ROOT = DATA_DIR / "food-101"
CHECKPOINT_DIR = ROOT / "checkpoints"
NUTRITION_PATH = DATA_DIR / "nutrition.json"
CLASS_INDEX_PATH = CHECKPOINT_DIR / "class_index.json"
HISTORY_PATH = CHECKPOINT_DIR / "history.json"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"

IMG_SIZE = 224
NUM_CLASSES = 101
BATCH_SIZE = 16
NUM_WORKERS = 2
EPOCHS = 12
FREEZE_EPOCHS = 3
LR_HEAD = 3e-3
LR_FINETUNE = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
SEED = 42
