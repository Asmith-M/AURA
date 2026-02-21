import os
from pathlib import Path

_requested_mode = os.getenv("AURA_MODE", "real").strip().lower()
if _requested_mode != "real":
    raise ValueError("Demo mode has been removed. Set AURA_MODE=real.")
AURA_MODE = "real"

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

RECEIVED_MODELS_DIR = BASE_DIR / "received_models"
XAI_REPORTS_DIR = BASE_DIR / "xai_reports"
DETECTOR_DIR = BASE_DIR / "detector"
GOLDEN_SET_DIR = BASE_DIR / "golden_set"

LEDGER_DB_PATH = BASE_DIR / "aura_ledger.db"
DETECTOR_MODEL_PATH = DETECTOR_DIR / "model.pkl"

# Prefer backend-local files first, then shared project data as fallback.
GOLDEN_SET_PATH_CANDIDATES = [
    GOLDEN_SET_DIR / "golden_test.npy",
    GOLDEN_SET_DIR / "golden_set.npy",
    PROJECT_ROOT / "data" / "golden_set.npy",
]
GOLDEN_LABELS_PATH_CANDIDATES = [
    GOLDEN_SET_DIR / "golden_labels.npy",
    PROJECT_ROOT / "data" / "golden_labels.npy",
]

ANOMALY_THRESHOLD = float(os.getenv("AURA_ANOMALY_THRESHOLD", "0.72"))
SHAP_SAMPLE_SIZE = int(os.getenv("AURA_SHAP_SAMPLE_SIZE", "20"))
SHAP_BACKGROUND_SIZE = int(os.getenv("AURA_SHAP_BACKGROUND_SIZE", "10"))
RANDOM_SEED = int(os.getenv("AURA_RANDOM_SEED", "42"))

for path in [RECEIVED_MODELS_DIR, XAI_REPORTS_DIR, DETECTOR_DIR, GOLDEN_SET_DIR]:
    path.mkdir(parents=True, exist_ok=True)
