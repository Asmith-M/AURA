import os
from typing import Dict, Any

# Legacy configuration module. Live AURA backend settings are defined in
# `aura_backend/config.py` and should be treated as the source of truth.
class Settings:
    # Database settings
    DATABASE_URL: str = "sqlite:///./transactions.db"
    
    # SHAP settings
    SHAP_SAMPLE_SIZE: int = 100
    SHAP_MAX_EVALS: int = 1000
    
    # Detection settings
    ANOMALY_THRESHOLD: float = 0.5
    ISOLATION_CONTRAMIN: float = 0.1
    
    # Golden set settings
    GOLDEN_SET_PATH: str = "./data/golden_set.npy"
    GOLDEN_SET_SIZE: int = 100
    
    # Model settings
    MODEL_INPUT_SIZE: int = 28 * 28  # For MNIST
    NUM_CLASSES: int = 10
    
    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Storage paths
    XAI_REPORTS_DIR: str = "./xai_reports"
    FINGERPRINTS_DIR: str = "./fingerprints"
    SENTINEL_MODELS_DIR: str = "./sentinel/received_models"
    TRAINING_DATA_DIR: str = "./data/train_data"

# Global settings instance
settings = Settings()
