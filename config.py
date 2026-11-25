
from pathlib import Path
import torch


# Project directories
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

CHECKPOINT_DIR = BASE_DIR / "models" / "checkpoints"


CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# Hardware
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# Initial experiment settings
RANDOM_SEED = 42

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360

BATCH_SIZE = 16
LEARNING_RATE = 0.001

DETECTION_EPOCHS = 20
RECOGNITION_EPOCHS = 30

print(f"Using device: {DEVICE}")
