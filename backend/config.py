import os
from pathlib import Path
import torch


class Config:
    # Flask config
    DEBUG = os.getenv('FLASK_DEBUG', False)
    UPLOAD_FOLDER = Path(__file__).parent / 'uploads'
    PREDICTIONS_FOLDER = Path(__file__).parent / 'predictions'
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size

    # Model config
    MODEL_PATH = os.getenv('MODEL_PATH', 'nnU-net/best_model_try1.pth')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    INPUT_CHANNELS = 4  # T1, T1c, T2, FLAIR
    OUTPUT_CHANNELS = 4  # Background, NCR/NET, ED, ET

    # Preprocessing config
    TARGET_SHAPE = (128, 128, 128)
    NORMALIZATION_METHOD = 'znorm'  # z-score normalization

    # Create folders if they don't exist
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_FOLDER.mkdir(parents=True, exist_ok=True)


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False