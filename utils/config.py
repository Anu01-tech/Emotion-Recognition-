import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Model Paths
MODEL_DIR = BASE_DIR / "models" / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Shared Classes (Unified mapped classes)
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
NUM_CLASSES = len(EMOTIONS)

# Training Hyperparameters
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4

# Audio Config
SAMPLE_RATE = 22050
DURATION = 3 # seconds
N_MFCC = 40

# Image Config
IMAGE_SIZE = (224, 224)

# Text Config
MAX_LEN = 128
TEXT_MODEL_NAME = "distilbert-base-uncased"
