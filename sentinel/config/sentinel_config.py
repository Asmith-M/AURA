from typing import Dict, Any
import os

class SentinelConfig:
    """Configuration for the Aura Sentinel system"""
    
    # API Configuration
    API_HOST: str = os.getenv("SENTINEL_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("SENTINEL_PORT", "8000"))
    API_LOG_LEVEL: str = os.getenv("SENTINEL_LOG_LEVEL", "info")
    
    # Storage Configuration
    MODELS_DIR: str = os.getenv("SENTINEL_MODELS_DIR", "./sentinel/received_models")
    REPORTS_DIR: str = os.getenv("SENTINEL_REPORTS_DIR", "./xai_reports")
    FINGERPRINTS_DIR: str = os.getenv("SENTINEL_FINGERPRINTS_DIR", "./fingerprints")
    
    # Analysis Configuration
    SHAP_SAMPLE_SIZE: int = int(os.getenv("SHAP_SAMPLE_SIZE", "50"))
    MAX_MODEL_SIZE: int = int(os.getenv("MAX_MODEL_SIZE", "100000000"))  # 100MB
    ANALYSIS_TIMEOUT: int = int(os.getenv("ANALYSIS_TIMEOUT", "300"))  # 5 minutes
    
    # Security Configuration
    ENABLE_MODEL_VALIDATION: bool = os.getenv("ENABLE_MODEL_VALIDATION", "true").lower() == "true"
    ENABLE_SHAP_ANALYSIS: bool = os.getenv("ENABLE_SHAP_ANALYSIS", "true").lower() == "true"
    ANOMALY_THRESHOLD: float = float(os.getenv("ANOMALY_THRESHOLD", "0.5"))
    
    # Database Configuration (for ledger integration in Sprint 5)
    LEDGER_API_URL: str = os.getenv("LEDGER_API_URL", "http://localhost:8001")
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get all configuration as dictionary"""
        return {
            'api_host': cls.API_HOST,
            'api_port': cls.API_PORT,
            'api_log_level': cls.API_LOG_LEVEL,
            'models_dir': cls.MODELS_DIR,
            'reports_dir': cls.REPORTS_DIR,
            'fingerprints_dir': cls.FINGERPRINTS_DIR,
            'shap_sample_size': cls.SHAP_SAMPLE_SIZE,
            'max_model_size': cls.MAX_MODEL_SIZE,
            'analysis_timeout': cls.ANALYSIS_TIMEOUT,
            'enable_model_validation': cls.ENABLE_MODEL_VALIDATION,
            'enable_shap_analysis': cls.ENABLE_SHAP_ANALYSIS,
            'anomaly_threshold': cls.ANOMALY_THRESHOLD,
            'ledger_api_url': cls.LEDGER_API_URL
        }

# Global config instance
sentinel_config = SentinelConfig()