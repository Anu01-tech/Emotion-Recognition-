import os
from pathlib import Path
import sys

# Add parent dir to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.config import RAW_DATA_DIR

def setup_data_directories():
    print("Setting up data directories...")
    
    fer_dir = RAW_DATA_DIR / "fer2013"
    ravdess_dir = RAW_DATA_DIR / "ravdess"
    
    fer_dir.mkdir(parents=True, exist_ok=True)
    ravdess_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Please place FER2013 dataset in: {fer_dir}")
    print(f"Please place RAVDESS dataset in: {ravdess_dir}")
    print("GoEmotions dataset will be downloaded automatically via HuggingFace datasets library during training.")
    print("\nDirectory setup complete.")

if __name__ == "__main__":
    setup_data_directories()
