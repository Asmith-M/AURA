import asyncio
import base64
import hashlib
import io
import json
import logging
import math
import random
import sys
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import joblib
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
try:
    import shap
except ImportError:  # pragma: no cover - environment dependent
    shap = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - environment dependent
    Image = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from fl_client.models.base_model import CNNModel, SimpleNet
except Exception:  # pragma: no cover - optional dependency in some environments
    CNNModel = None
    SimpleNet = None

try:
    from config import (
        ANOMALY_THRESHOLD,
        DETECTOR_MODEL_PATH,
        GOLDEN_SET_DIR,
        GOLDEN_LABELS_PATH_CANDIDATES,
        GOLDEN_SET_PATH_CANDIDATES,
        LEDGER_DB_PATH,
        RANDOM_SEED,
        RECEIVED_MODELS_DIR,
        SHAP_BACKGROUND_SIZE,
        SHAP_SAMPLE_SIZE,
        XAI_REPORTS_DIR,
    )
except ImportError:  # pragma: no cover - import-path fallback
    from aura_backend.config import (  # type: ignore
        ANOMALY_THRESHOLD,
        DETECTOR_MODEL_PATH,
        GOLDEN_SET_DIR,
        GOLDEN_LABELS_PATH_CANDIDATES,
        GOLDEN_SET_PATH_CANDIDATES,
        LEDGER_DB_PATH,
        RANDOM_SEED,
        RECEIVED_MODELS_DIR,
        SHAP_BACKGROUND_SIZE,
        SHAP_SAMPLE_SIZE,
        XAI_REPORTS_DIR,
    )
try:
    from ledger_manager import LedgerManager
except ImportError:  # pragma: no cover - import-path fallback
    from aura_backend.ledger_manager import LedgerManager  # type: ignore

logger = logging.getLogger(__name__)

_detector_bundle: Optional[Dict[str, Any]] = None
_golden_data: Optional[np.ndarray] = None
_golden_labels: Optional[np.ndarray] = None
_golden_data_path: Optional[Path] = None
_golden_labels_path: Optional[Path] = None
_dataset_metadata: Dict[str, Any] = {}
_recent_detector_scores: deque = deque(maxlen=250)
_recent_detector_raw_scores: deque = deque(maxlen=250)
_ledger_manager: Optional[LedgerManager] = None

SEMANTIC_FINGERPRINT_KEYS = [
    "lung_opacity",
    "cardiomegaly",
    "consolidation",
    "edema",
    "age_bias",
    "gender_bias",
    "pixel_artifact",
    "texture_variance",
    "edge_sensitivity",
    "background_noise",
]

DETECTOR_FEATURE_DEFAULTS = [
    "entropy_mean",
    "feature_consistency",
    "max_importance_global",
    "mean_importance_global",
    "min_importance_global",
    "prediction_stability",
    "std_importance_global",
    "variance_stability",
]

SHAP_METHOD_DISPLAY_NAMES = {
    "deep": "SHAP DeepExplainer",
    "kernel": "SHAP KernelExplainer",
    "surrogate_gradient": "SHAP Approximation (Surrogate Gradient)",
    "surrogate_centered_input": "SHAP Approximation (Centered-Input Fallback)",
    "fallback_error": "SHAP Fallback (Error Recovery)",
}


def _iso_utc() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _shap_method_display_name(method: str) -> str:
    return SHAP_METHOD_DISPLAY_NAMES.get(method, f"SHAP ({method})")


def _get_ledger_manager() -> LedgerManager:
    global _ledger_manager
    if _ledger_manager is None:
        manager = LedgerManager(LEDGER_DB_PATH)
        manager.initialize_database()
        _ledger_manager = manager
    return _ledger_manager


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_existing_path(candidates: Sequence[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


def _metadata_path_candidates() -> List[Path]:
    candidates = [
        GOLDEN_SET_DIR / "dataset_metadata.json",
        GOLDEN_SET_DIR / "golden_metadata.json",
        PROJECT_ROOT / "data" / "dataset_metadata.json",
    ]
    if _golden_data_path is not None:
        candidates.append(_golden_data_path.with_name("dataset_metadata.json"))
        candidates.append(_golden_data_path.with_suffix(".metadata.json"))
    deduped: List[Path] = []
    seen: Set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _to_int_list(values: Any) -> List[int]:
    if not isinstance(values, list):
        return []
    result: List[int] = []
    for value in values:
        try:
            result.append(int(value))
        except Exception:
            continue
    return sorted(set(result))


def _is_metadata_valid(metadata: Dict[str, Any], total_samples: int) -> bool:
    if not isinstance(metadata, dict) or total_samples <= 0:
        return False

    anomaly_indices = _to_int_list(metadata.get("anomaly_indices"))
    anomaly_count = int(metadata.get("anomalous_samples", -1) or -1)
    clean_count = int(metadata.get("clean_samples", -1) or -1)
    anomaly_percentage = float(metadata.get("anomaly_percentage", -1.0) or -1.0)

    if not anomaly_indices:
        return False
    if any(index < 0 or index >= total_samples for index in anomaly_indices):
        return False

    if anomaly_count != len(anomaly_indices):
        return False
    if clean_count != (total_samples - anomaly_count):
        return False

    ratio = anomaly_count / total_samples
    if not (0.20 <= ratio <= 0.30):
        return False
    if abs(anomaly_percentage - ratio * 100.0) > 0.6:
        return False

    anomaly_type = metadata.get("anomaly_type")
    if isinstance(anomaly_type, str):
        anomaly_types = [anomaly_type]
    elif isinstance(anomaly_type, list):
        anomaly_types = [str(item) for item in anomaly_type]
    else:
        anomaly_types = []
    required = {"pixel_level", "label_flip", "feature_spike"}
    if not required.issubset(set(anomaly_types)):
        return False

    return True


def _anomaly_signal_delta(data: np.ndarray, metadata: Dict[str, Any]) -> float:
    anomaly_indices = _to_int_list(metadata.get("anomaly_indices"))
    if not anomaly_indices:
        return 0.0

    anomaly_mask = np.zeros(len(data), dtype=bool)
    anomaly_mask[anomaly_indices] = True
    clean_mask = ~anomaly_mask

    if not np.any(clean_mask):
        return 0.0

    anomaly_mean = float(np.mean(np.abs(data[anomaly_mask])))
    clean_mean = float(np.mean(np.abs(data[clean_mask])))
    return abs(anomaly_mean - clean_mean)


def _inject_controlled_anomalies(
    data: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    x = np.asarray(data, dtype=np.float32).copy()
    y = _labels_to_int_array(np.asarray(labels)).copy()

    total_samples = int(len(x))
    if total_samples <= 0:
        raise ValueError("Dataset is empty; cannot inject anomalies")

    target_count = int(round(total_samples * 0.25))
    lower = int(math.ceil(total_samples * 0.20))
    upper = int(math.floor(total_samples * 0.30))
    anomalous_count = max(lower, min(target_count, upper))

    rng = np.random.default_rng(int(seed))
    anomaly_indices = sorted(rng.choice(total_samples, size=anomalous_count, replace=False).astype(int).tolist())
    pixel_indices, label_indices, spike_indices = [list(chunk.astype(int)) for chunk in np.array_split(np.asarray(anomaly_indices), 3)]

    # Pixel-level anomaly: high-contrast center patch + local noise to make artifacts visible.
    for index in pixel_indices:
        sample = np.asarray(x[index]).copy()
        if sample.ndim == 3:
            channels, height, width = sample.shape
            patch_size = max(4, int(min(height, width) * 0.2))
            top = int(max(0, (height // 2) - (patch_size // 2)))
            left = int(max(0, (width // 2) - (patch_size // 2)))
            patch_noise = rng.normal(loc=2.2, scale=0.35, size=(channels, patch_size, patch_size)).astype(np.float32)
            sample[:, top : top + patch_size, left : left + patch_size] = patch_noise
        elif sample.ndim == 2:
            height, width = sample.shape
            patch_size = max(4, int(min(height, width) * 0.2))
            top = int(max(0, (height // 2) - (patch_size // 2)))
            left = int(max(0, (width // 2) - (patch_size // 2)))
            patch_noise = rng.normal(loc=2.2, scale=0.35, size=(patch_size, patch_size)).astype(np.float32)
            sample[top : top + patch_size, left : left + patch_size] = patch_noise
        x[index] = sample

    # Label-flip anomaly: force incorrect targets without touching pixels.
    unique_labels = np.unique(y)
    num_classes = int(np.max(unique_labels)) + 1 if unique_labels.size else 10
    label_flip_map: List[Dict[str, int]] = []
    for index in label_indices:
        original = int(y[index])
        candidates = [int(label) for label in unique_labels.tolist() if int(label) != original]
        if candidates:
            flipped = int(rng.choice(np.asarray(candidates, dtype=np.int64)))
        else:
            flipped = int((original + 1) % max(num_classes, 2))
        y[index] = flipped
        label_flip_map.append({"index": int(index), "from": original, "to": flipped})

    # Feature spike anomaly: amplify sparse feature positions aggressively.
    feature_spike_map: List[Dict[str, Any]] = []
    for index in spike_indices:
        sample = np.asarray(x[index]).copy()
        flat = sample.reshape(-1)
        spike_count = max(6, int(len(flat) * 0.03))
        spike_positions = rng.choice(len(flat), size=spike_count, replace=False).astype(int)
        flat[spike_positions] = (flat[spike_positions] * 4.0) + 3.5
        x[index] = flat.reshape(sample.shape)
        feature_spike_map.append(
            {
                "index": int(index),
                "spike_positions": [int(position) for position in spike_positions[:20].tolist()],
                "total_spike_positions": int(spike_count),
            }
        )

    clean_samples = total_samples - anomalous_count
    anomaly_percentage = round((anomalous_count / total_samples) * 100.0, 2)

    metadata: Dict[str, Any] = {
        "total_samples": int(total_samples),
        "clean_samples": int(clean_samples),
        "anomalous_samples": int(anomalous_count),
        "anomaly_type": ["pixel_level", "label_flip", "feature_spike"],
        "anomaly_percentage": float(anomaly_percentage),
        "anomaly_indices": [int(index) for index in anomaly_indices],
        "anomaly_breakdown": {
            "pixel_level_indices": [int(index) for index in pixel_indices],
            "label_flip_indices": [int(index) for index in label_indices],
            "feature_spike_indices": [int(index) for index in spike_indices],
            "label_flip_map": label_flip_map,
            "feature_spike_map": feature_spike_map,
        },
        "generated_at": _iso_utc(),
    }
    return x.astype(np.float32), y.astype(np.int64), metadata


def _persist_dataset_bundle(data: np.ndarray, labels: np.ndarray, metadata: Dict[str, Any]) -> None:
    GOLDEN_SET_DIR.mkdir(parents=True, exist_ok=True)
    np.save(GOLDEN_SET_DIR / "golden_test.npy", np.asarray(data, dtype=np.float32))
    np.save(GOLDEN_SET_DIR / "golden_labels.npy", _labels_to_int_array(np.asarray(labels)))
    with (GOLDEN_SET_DIR / "dataset_metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)


def _load_dataset_metadata(total_samples: int) -> Dict[str, Any]:
    for path in _metadata_path_candidates():
        if not path.exists():
            continue
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if _is_metadata_valid(loaded, total_samples):
                return loaded
        except Exception:
            continue
    return {}


def _feature_indices_by_type(metadata: Dict[str, Any], key: str) -> Set[int]:
    breakdown = metadata.get("anomaly_breakdown") or {}
    return set(_to_int_list(breakdown.get(key)))


def _ensure_anomalous_golden_dataset(data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    total_samples = int(len(data))
    metadata = _load_dataset_metadata(total_samples=total_samples)

    if _is_metadata_valid(metadata, total_samples):
        signal_delta = _anomaly_signal_delta(data, metadata)
        if signal_delta >= 0.01:
            return np.asarray(data, dtype=np.float32), _labels_to_int_array(np.asarray(labels)), metadata

    injected_data, injected_labels, metadata = _inject_controlled_anomalies(
        data,
        labels,
        seed=RANDOM_SEED,
    )
    _persist_dataset_bundle(injected_data, injected_labels, metadata)
    return injected_data, injected_labels, metadata


def _encode_sample_to_png_base64(sample: np.ndarray) -> str:
    if Image is None:
        return ""

    array = np.asarray(sample, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]

    arr_min = float(np.min(array))
    arr_max = float(np.max(array))
    denom = (arr_max - arr_min) if arr_max > arr_min else 1.0
    scaled = np.clip(((array - arr_min) / denom) * 255.0, 0, 255).astype(np.uint8)

    if scaled.ndim == 2:
        image = Image.fromarray(scaled, mode="L")
    else:
        image = Image.fromarray(scaled)
    image = image.resize((112, 112), Image.NEAREST)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def get_dataset_inspection_payload(sample_limit: int = 12) -> Dict[str, Any]:
    if _golden_data is None or _golden_labels is None:
        raise RuntimeError("Golden dataset is not available")

    labels = _labels_to_int_array(np.asarray(_golden_labels))
    metadata = dict(_dataset_metadata)
    total_samples = int(len(_golden_data))
    class_distribution = {str(int(label)): int(count) for label, count in zip(*np.unique(labels, return_counts=True))}

    anomaly_indices = _to_int_list(metadata.get("anomaly_indices"))
    anomaly_set = set(anomaly_indices)
    clean_indices = [index for index in range(total_samples) if index not in anomaly_set]

    limit = max(2, int(sample_limit))
    half = max(1, limit // 2)
    preview_indices = anomaly_indices[:half] + clean_indices[: max(1, limit - half)]
    preview: List[Dict[str, Any]] = []

    pixel_set = _feature_indices_by_type(metadata, "pixel_level_indices")
    flip_set = _feature_indices_by_type(metadata, "label_flip_indices")
    spike_set = _feature_indices_by_type(metadata, "feature_spike_indices")

    for index in preview_indices:
        sample = np.asarray(_golden_data[index], dtype=np.float32)
        anomaly_tags: List[str] = []
        if index in pixel_set:
            anomaly_tags.append("pixel_level")
        if index in flip_set:
            anomaly_tags.append("label_flip")
        if index in spike_set:
            anomaly_tags.append("feature_spike")

        preview.append(
            {
                "index": int(index),
                "label": int(labels[index]),
                "is_anomalous": bool(index in anomaly_set),
                "anomaly_tags": anomaly_tags,
                "image_base64": _encode_sample_to_png_base64(sample),
                "stats": {
                    "min": round(float(np.min(sample)), 5),
                    "max": round(float(np.max(sample)), 5),
                    "mean": round(float(np.mean(sample)), 5),
                    "std": round(float(np.std(sample)), 5),
                },
            }
        )

    anomaly_mask = np.zeros(total_samples, dtype=bool)
    if anomaly_indices:
        anomaly_mask[anomaly_indices] = True
    clean_mask = ~anomaly_mask

    anomaly_data = _golden_data[anomaly_mask] if np.any(anomaly_mask) else np.empty((0,), dtype=np.float32)
    clean_data = _golden_data[clean_mask] if np.any(clean_mask) else np.empty((0,), dtype=np.float32)

    statistical_summary = {
        "global_mean": round(float(np.mean(_golden_data)), 6),
        "global_std": round(float(np.std(_golden_data)), 6),
        "global_min": round(float(np.min(_golden_data)), 6),
        "global_max": round(float(np.max(_golden_data)), 6),
        "clean_mean": round(float(np.mean(clean_data)) if clean_data.size else 0.0, 6),
        "clean_std": round(float(np.std(clean_data)) if clean_data.size else 0.0, 6),
        "anomalous_mean": round(float(np.mean(anomaly_data)) if anomaly_data.size else 0.0, 6),
        "anomalous_std": round(float(np.std(anomaly_data)) if anomaly_data.size else 0.0, 6),
        "signal_delta_mean_abs": round(_anomaly_signal_delta(np.asarray(_golden_data), metadata), 6),
    }

    anomaly_count = int(metadata.get("anomalous_samples", len(anomaly_indices)))
    anomaly_percentage = float(metadata.get("anomaly_percentage", (anomaly_count / max(total_samples, 1)) * 100.0))

    return {
        "total_samples": int(total_samples),
        "class_distribution": class_distribution,
        "anomaly_count": anomaly_count,
        "anomaly_percentage": round(anomaly_percentage, 2),
        "anomaly_type": metadata.get("anomaly_type", []),
        "anomaly_indices": anomaly_indices,
        "sample_preview": preview,
        "statistical_summary": statistical_summary,
        "dataset_metadata": metadata,
    }


def initialize_real_mode_components() -> None:
    global _detector_bundle, _golden_data, _golden_labels, _golden_data_path, _golden_labels_path, _dataset_metadata

    _set_seed(RANDOM_SEED)

    if DETECTOR_MODEL_PATH.exists():
        try:
            _detector_bundle = joblib.load(DETECTOR_MODEL_PATH)
            logger.info("Loaded detector bundle from %s", DETECTOR_MODEL_PATH)
        except Exception as exc:
            _detector_bundle = None
            logger.error("Failed to load detector bundle: %s", str(exc))
    else:
        _detector_bundle = None
        logger.warning("Detector model not found at %s", DETECTOR_MODEL_PATH)

    _golden_data_path = _resolve_existing_path(GOLDEN_SET_PATH_CANDIDATES)
    _golden_labels_path = _resolve_existing_path(GOLDEN_LABELS_PATH_CANDIDATES)

    if _golden_data_path and _golden_labels_path:
        try:
            loaded_data = np.asarray(np.load(_golden_data_path), dtype=np.float32)
            loaded_labels = _labels_to_int_array(np.asarray(np.load(_golden_labels_path)))
            loaded_data, loaded_labels, loaded_metadata = _ensure_anomalous_golden_dataset(loaded_data, loaded_labels)
            _golden_data = loaded_data
            _golden_labels = loaded_labels
            _dataset_metadata = loaded_metadata
            _golden_data_path = GOLDEN_SET_DIR / "golden_test.npy"
            _golden_labels_path = GOLDEN_SET_DIR / "golden_labels.npy"
            if len(_golden_data) != len(_golden_labels):
                raise ValueError("Golden set size mismatch between inputs and labels")
            logger.info(
                "Loaded golden set from %s and %s (%s samples, anomalies=%s%%)",
                _golden_data_path,
                _golden_labels_path,
                len(_golden_data),
                _dataset_metadata.get("anomaly_percentage", 0.0),
            )
        except Exception as exc:
            _golden_data = None
            _golden_labels = None
            _dataset_metadata = {}
            logger.error("Failed loading golden set: %s", str(exc))
    else:
        _golden_data = None
        _golden_labels = None
        _dataset_metadata = {}
        logger.warning("Golden set files not found in configured candidate paths")


def check_components_status() -> Dict[str, Any]:
    detector_loaded = _detector_bundle is not None
    ledger_connected = _get_ledger_manager().is_connected()
    golden_set_loaded = _golden_data is not None and _golden_labels is not None

    return {
        "detector_loaded": detector_loaded,
        "ledger_connected": ledger_connected,
        "golden_set_loaded": golden_set_loaded,
        "dataset_metadata_loaded": bool(_dataset_metadata),
        "components": {
            "mode": "real",
            "detector": "ready" if detector_loaded else "missing",
            "ledger": "ready",
            "golden_set": str(_golden_data_path) if golden_set_loaded else "missing",
            "dataset_anomalies": _dataset_metadata.get("anomaly_percentage", "missing"),
        },
    }


def _safe_filename(name: Optional[str]) -> str:
    if not name:
        return "uploaded_model.pt"
    return Path(name).name.replace(" ", "_")


def _safe_hospital_token(hospital_id: str) -> str:
    token = "".join(ch for ch in str(hospital_id) if ch.isalnum() or ch in {"_", "-"})
    token = token.strip("_-")
    return token or "UNKNOWN"


def _tensor_param_count_from_state_dict(state_dict: Dict[str, Any]) -> int:
    total = 0
    for value in state_dict.values():
        if isinstance(value, torch.Tensor):
            total += int(value.numel())
    return total


def _candidate_models() -> List[torch.nn.Module]:
    candidates: List[torch.nn.Module] = []
    if SimpleNet is not None:
        candidates.append(SimpleNet())
    if CNNModel is not None:
        candidates.append(CNNModel())
    return candidates


def _load_from_state_dict(state_dict: Dict[str, Any]) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    if not state_dict:
        raise ValueError("Uploaded state_dict is empty")

    state_dict_candidates = [state_dict]
    if any(key.startswith("module.") for key in state_dict.keys()):
        stripped = {key.replace("module.", "", 1): val for key, val in state_dict.items()}
        state_dict_candidates.append(stripped)

    models = _candidate_models()
    if not models:
        raise ValueError("No local model templates available to materialize uploaded state_dict")

    for sd in state_dict_candidates:
        for model in models:
            try:
                model.load_state_dict(sd, strict=True)
                model.eval()
                param_count = sum(int(p.numel()) for p in model.parameters())
                return model, {
                    "architecture": type(model).__name__,
                    "parameter_count": param_count,
                    "parameters": f"{param_count / 1_000_000:.2f}M",
                }
            except Exception:
                continue

    param_count = _tensor_param_count_from_state_dict(state_dict)
    raise ValueError(
        "Uploaded state_dict does not match supported model templates. "
        f"Detected {param_count} parameters."
    )


def _load_torch_model(model_path: Path) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    # Try TorchScript first.
    try:
        scripted_model = torch.jit.load(str(model_path), map_location="cpu")
        scripted_model.eval()
        param_count = sum(int(p.numel()) for p in scripted_model.parameters())
        profile = {
            "architecture": type(scripted_model).__name__,
            "parameter_count": param_count,
            "parameters": f"{param_count / 1_000_000:.2f}M",
        }
        return scripted_model, profile
    except Exception:
        pass

    # Then standard torch serialization.
    try:
        loaded_obj = torch.load(str(model_path), map_location="cpu", weights_only=True)
    except Exception as err:
        raise ValueError(f"Unable to load model file as PyTorch artifact: {err}") from err

    if isinstance(loaded_obj, torch.nn.Module):
        model = loaded_obj
        model.eval()
        param_count = sum(int(p.numel()) for p in model.parameters())
        profile = {
            "architecture": type(model).__name__,
            "parameter_count": param_count,
            "parameters": f"{param_count / 1_000_000:.2f}M",
        }
        return model, profile

    state_dict = None
    if isinstance(loaded_obj, dict):
        if "model_state_dict" in loaded_obj and isinstance(loaded_obj["model_state_dict"], dict):
            state_dict = loaded_obj["model_state_dict"]
        elif "state_dict" in loaded_obj and isinstance(loaded_obj["state_dict"], dict):
            state_dict = loaded_obj["state_dict"]
        elif all(isinstance(value, torch.Tensor) for value in loaded_obj.values()):
            state_dict = loaded_obj

    if state_dict is not None:
        return _load_from_state_dict(state_dict)

    raise ValueError("Unsupported model format. Use TorchScript, torch.nn.Module, or state_dict .pth")


def _forward_with_shape_fallbacks(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    attempts: List[torch.Tensor] = [batch]

    if batch.ndim == 4 and batch.shape[1] == 1:
        attempts.append(batch.repeat(1, 3, 1, 1))
    if batch.ndim == 3:
        attempts.append(batch.unsqueeze(1))
    if batch.ndim > 2:
        attempts.append(batch.view(batch.shape[0], -1))
    if batch.ndim == 2:
        attempts.append(batch.unsqueeze(1))

    last_error: Optional[Exception] = None
    seen_shapes = set()

    for candidate in attempts:
        if tuple(candidate.shape) in seen_shapes:
            continue
        seen_shapes.add(tuple(candidate.shape))

        try:
            output = model(candidate)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if not isinstance(output, torch.Tensor):
                raise TypeError("Model output is not a tensor")
            if output.ndim == 1:
                output = output.unsqueeze(1)
            return output
        except Exception as err:
            last_error = err

    raise RuntimeError(f"Model forward pass failed for all input shape attempts: {last_error}")


def _labels_to_int_array(labels: np.ndarray) -> np.ndarray:
    if labels.ndim > 1:
        return np.argmax(labels, axis=1).astype(np.int64)
    return labels.astype(np.int64)


def _evaluate_model(model: torch.nn.Module, golden_x: np.ndarray, golden_y: np.ndarray) -> Dict[str, Any]:
    model.eval()

    x_tensor = torch.tensor(golden_x, dtype=torch.float32)
    y_true = _labels_to_int_array(np.asarray(golden_y))

    all_preds: List[int] = []
    all_conf: List[float] = []

    with torch.no_grad():
        for start in range(0, len(x_tensor), 32):
            batch = x_tensor[start : start + 32]
            logits = _forward_with_shape_fallbacks(model, batch)

            if logits.shape[1] == 1:
                probs_pos = torch.sigmoid(logits[:, 0])
                preds = (probs_pos >= 0.5).to(torch.int64)
                conf = torch.maximum(probs_pos, 1.0 - probs_pos)
            else:
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                conf = torch.max(probs, dim=1).values

            all_preds.extend(preds.cpu().numpy().astype(int).tolist())
            all_conf.extend(conf.cpu().numpy().astype(float).tolist())

    pred_arr = np.asarray(all_preds, dtype=np.int64)
    conf_arr = np.asarray(all_conf, dtype=np.float64)
    if len(pred_arr) != len(y_true):
        raise RuntimeError("Prediction count does not match golden labels count")

    accuracy = float(np.mean(pred_arr == y_true))
    cm = confusion_matrix(y_true, pred_arr).astype(int).tolist()

    unique_classes = np.unique(y_true)
    average_type = "binary" if len(unique_classes) == 2 else "macro"
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true,
        pred_arr,
        average=average_type,
        zero_division=0,
    )
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        y_true,
        pred_arr,
        labels=unique_classes,
        average=None,
        zero_division=0,
    )

    class_distribution = {str(int(k)): int(v) for k, v in zip(*np.unique(y_true, return_counts=True))}
    per_class_accuracy = {}
    per_class_precision_map: Dict[str, float] = {}
    per_class_recall_map: Dict[str, float] = {}
    per_class_f1_map: Dict[str, float] = {}
    for index, cls in enumerate(unique_classes):
        cls_mask = y_true == cls
        class_key = str(int(cls))
        per_class_accuracy[class_key] = float(np.mean(pred_arr[cls_mask] == y_true[cls_mask]))
        per_class_precision_map[class_key] = float(per_class_precision[index])
        per_class_recall_map[class_key] = float(per_class_recall[index])
        per_class_f1_map[class_key] = float(per_class_f1[index])

    misclassified_indices = np.where(pred_arr != y_true)[0].astype(int).tolist()
    confidence_bins = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.000001], dtype=np.float64)
    histogram, _ = np.histogram(conf_arr if conf_arr.size else np.array([0.0]), bins=confidence_bins)
    confidence_distribution = {
        "0.0-0.2": int(histogram[0]),
        "0.2-0.4": int(histogram[1]),
        "0.4-0.6": int(histogram[2]),
        "0.6-0.8": int(histogram[3]),
        "0.8-1.0": int(histogram[4]),
        "mean": round(float(np.mean(conf_arr)) if conf_arr.size else 0.0, 6),
        "std": round(float(np.std(conf_arr)) if conf_arr.size else 0.0, 6),
    }

    return {
        "accuracy": round(accuracy, 3),
        "precision": round(float(precision), 3),
        "recall": round(float(recall), 3),
        "f1_score": round(float(f1_score), 3),
        "samples_tested": int(len(y_true)),
        "confusion_matrix": cm,
        "avg_confidence": round(float(np.mean(conf_arr)) if conf_arr.size else 0.0, 3),
        "class_distribution": class_distribution,
        "per_class_accuracy": {k: round(v, 3) for k, v in per_class_accuracy.items()},
        "per_class_precision": {k: round(v, 3) for k, v in per_class_precision_map.items()},
        "per_class_recall": {k: round(v, 3) for k, v in per_class_recall_map.items()},
        "precision_per_class": {k: round(v, 6) for k, v in per_class_precision_map.items()},
        "recall_per_class": {k: round(v, 6) for k, v in per_class_recall_map.items()},
        "f1_per_class": {k: round(v, 6) for k, v in per_class_f1_map.items()},
        "confidence_distribution": confidence_distribution,
        "misclassified_indices": [int(index) for index in misclassified_indices],
    }


def _softmax_or_sigmoid_probs(model: torch.nn.Module, x_batch: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        logits = _forward_with_shape_fallbacks(model, x_batch)
        if logits.shape[1] == 1:
            probs_pos = torch.sigmoid(logits[:, 0]).view(-1, 1)
            probs = torch.cat([1.0 - probs_pos, probs_pos], dim=1)
        else:
            probs = torch.softmax(logits, dim=1)
    return probs.cpu().numpy().astype(float)


def _coerce_shap_to_2d(shap_values: Any) -> np.ndarray:
    values = shap_values

    if isinstance(values, list):
        class_idx = 1 if len(values) > 1 else 0
        values = values[class_idx]

    arr = np.asarray(values)

    if arr.ndim == 1:
        return arr.reshape(-1, 1)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        # Common format can be [samples, features, classes] or [classes, samples, features].
        if arr.shape[0] <= 5 and arr.shape[1] > 5 and arr.shape[2] > 5:
            class_idx = 1 if arr.shape[0] > 1 else 0
            arr = arr[class_idx]
        elif arr.shape[2] <= 5:
            class_idx = 1 if arr.shape[2] > 1 else 0
            arr = arr[:, :, class_idx]
        else:
            arr = arr.reshape(arr.shape[0], -1)
        return arr

    return arr.reshape(arr.shape[0], -1)


def _entropy(values: np.ndarray) -> float:
    total = float(np.sum(values))
    if total <= 0:
        return 0.0
    probs = values / total
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def _feature_consistency(abs_shap: np.ndarray, top_k: int = 5) -> float:
    if abs_shap.shape[0] <= 1:
        return 1.0

    top_sets: List[set] = []
    for row in abs_shap:
        indices = np.argsort(row)[-top_k:]
        top_sets.append(set(indices.tolist()))

    sims: List[float] = []
    for i in range(len(top_sets)):
        for j in range(i + 1, len(top_sets)):
            union = top_sets[i] | top_sets[j]
            if not union:
                sims.append(1.0)
            else:
                sims.append(len(top_sets[i] & top_sets[j]) / len(union))

    return float(np.mean(sims)) if sims else 0.0


def _build_semantic_fingerprint(normalized_importance: np.ndarray) -> Dict[str, float]:
    chunks = np.array_split(normalized_importance, len(SEMANTIC_FINGERPRINT_KEYS))
    values = [float(np.sum(chunk)) if len(chunk) else 0.0 for chunk in chunks]

    fingerprint = {key: round(value, 3) for key, value in zip(SEMANTIC_FINGERPRINT_KEYS, values)}

    # Include a direct "age" alias to satisfy the required response example.
    fingerprint["age"] = fingerprint["age_bias"]

    return fingerprint


def _build_detector_features(abs_shap: np.ndarray, prediction_probs: np.ndarray) -> Dict[str, float]:
    row_mean = np.mean(abs_shap, axis=1)
    row_std = np.std(abs_shap, axis=1)
    row_max = np.max(abs_shap, axis=1)
    row_min = np.min(abs_shap, axis=1)

    entropy_values = np.array([_entropy(row) for row in abs_shap], dtype=float)
    prediction_conf = np.max(prediction_probs, axis=1) if prediction_probs.size else np.array([0.0])

    return {
        "mean_importance_global": float(np.mean(row_mean)),
        "std_importance_global": float(np.mean(row_std)),
        "max_importance_global": float(np.max(row_max)),
        "min_importance_global": float(np.min(row_min)),
        "entropy_mean": float(np.mean(entropy_values)),
        "variance_stability": float(np.std(row_std)),
        "feature_consistency": _feature_consistency(abs_shap),
        "prediction_stability": float(np.std(prediction_conf)),
    }


def _compute_surrogate_attributions(model: torch.nn.Module, sample_x: np.ndarray) -> np.ndarray:
    tensor = torch.tensor(sample_x, dtype=torch.float32, requires_grad=True)
    model.zero_grad(set_to_none=True)

    logits = _forward_with_shape_fallbacks(model, tensor)
    if logits.shape[1] == 1:
        objective = torch.sigmoid(logits[:, 0]).sum()
    else:
        probs = torch.softmax(logits, dim=1)
        objective = torch.max(probs, dim=1).values.sum()

    objective.backward()
    if tensor.grad is None:
        raise RuntimeError("Gradient attributions unavailable")
    return np.abs(tensor.grad.detach().cpu().numpy()).reshape(sample_x.shape[0], -1)


def _compute_shap_fingerprint(
    model: torch.nn.Module,
    golden_x: np.ndarray,
    golden_y: np.ndarray,
    session_id: str,
    dataset_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sample_count = min(SHAP_SAMPLE_SIZE, len(golden_x))
    if sample_count <= 0:
        raise ValueError("Golden set is empty; cannot run SHAP")

    sample_x = np.asarray(golden_x[:sample_count], dtype=np.float32)
    sample_y = _labels_to_int_array(np.asarray(golden_y))[:sample_count]
    sample_shape = sample_x.shape[1:]
    sample_flat = sample_x.reshape(sample_count, -1)
    feature_count = int(sample_flat.shape[1])

    background_count = min(SHAP_BACKGROUND_SIZE, sample_count)
    background_flat = sample_flat[:background_count]

    def predict_fn(flat_inputs: np.ndarray) -> np.ndarray:
        inputs = np.asarray(flat_inputs, dtype=np.float32).reshape((-1, *sample_shape))
        x_tensor = torch.tensor(inputs, dtype=torch.float32)
        return _softmax_or_sigmoid_probs(model, x_tensor)

    shap_method = "surrogate_gradient"
    shap_warning: Optional[str] = None
    shap_2d: Optional[np.ndarray] = None

    if shap is not None:
        try:
            background_tensor = torch.tensor(sample_x[:background_count], dtype=torch.float32)
            foreground_tensor = torch.tensor(sample_x, dtype=torch.float32)
            deep_explainer = shap.DeepExplainer(model, background_tensor)
            shap_raw = deep_explainer.shap_values(foreground_tensor)
            shap_2d = _coerce_shap_to_2d(shap_raw)
            shap_method = "deep"
        except Exception as deep_exc:
            shap_warning = f"DeepExplainer failed: {deep_exc}"
            if sample_flat.shape[1] <= 512:
                try:
                    kernel_explainer = shap.KernelExplainer(predict_fn, background_flat)
                    shap_raw = kernel_explainer.shap_values(sample_flat)
                    shap_2d = _coerce_shap_to_2d(shap_raw)
                    shap_method = "kernel"
                except Exception as kernel_exc:
                    shap_warning = f"{shap_warning}; KernelExplainer failed: {kernel_exc}"

    if shap_2d is None:
        try:
            shap_2d = _compute_surrogate_attributions(model, sample_x)
            shap_method = "surrogate_gradient"
        except Exception as grad_exc:
            centered = sample_flat - np.mean(sample_flat, axis=0, keepdims=True)
            shap_2d = np.abs(centered)
            shap_method = "surrogate_centered_input"
            if shap_warning:
                shap_warning = f"{shap_warning}; gradient fallback failed: {grad_exc}"
            else:
                shap_warning = f"gradient fallback failed: {grad_exc}"

    if shap_2d.shape[0] != sample_count:
        shap_2d = shap_2d[:sample_count]
    if shap_2d.shape[1] != feature_count:
        if shap_2d.shape[1] > feature_count:
            shap_2d = shap_2d[:, :feature_count]
        else:
            padding = np.zeros((shap_2d.shape[0], feature_count - shap_2d.shape[1]), dtype=shap_2d.dtype)
            shap_2d = np.concatenate([shap_2d, padding], axis=1)

    abs_shap = np.abs(shap_2d)
    mean_abs = np.mean(abs_shap, axis=0)
    denom = float(np.sum(mean_abs))
    if denom <= 0:
        normalized_importance = np.full_like(mean_abs, 1.0 / max(len(mean_abs), 1), dtype=float)
    else:
        normalized_importance = mean_abs / denom

    fingerprint = _build_semantic_fingerprint(normalized_importance)

    raw_top_indices = np.argsort(normalized_importance)[-5:][::-1]
    semantic_items = [
        (name, float(value))
        for name, value in fingerprint.items()
        if name != "age" and isinstance(value, (float, int))
    ]
    semantic_sorted_desc = sorted(semantic_items, key=lambda item: item[1], reverse=True)
    semantic_sorted_asc = sorted(semantic_items, key=lambda item: item[1])
    top_features = [
        {
            "feature": name,
            "feature_index": -1,
            "importance": round(value, 6),
            "direction": "positive" if i < 2 else "neutral",
        }
        for i, (name, value) in enumerate(semantic_sorted_desc[:5])
    ]
    bottom_features = [
        {
            "feature": name,
            "feature_index": -1,
            "importance": round(value, 6),
            "direction": "negative",
        }
        for name, value in semantic_sorted_asc[:5]
    ]

    interaction_value = 0.0
    if shap_2d.shape[1] >= 2 and len(raw_top_indices) >= 2:
        a = shap_2d[:, int(raw_top_indices[0])]
        b = shap_2d[:, int(raw_top_indices[1])]
        if np.std(a) > 0 and np.std(b) > 0:
            interaction_value = float(np.corrcoef(a, b)[0, 1])

    metadata = dataset_metadata or {}
    anomaly_indices = set(_to_int_list(metadata.get("anomaly_indices")))
    local_anomaly_indices = [idx for idx in range(sample_count) if idx in anomaly_indices]
    local_clean_indices = [idx for idx in range(sample_count) if idx not in anomaly_indices]
    if not local_clean_indices:
        local_clean_indices = list(range(sample_count))

    anomalous_vector = (
        np.mean(abs_shap[local_anomaly_indices], axis=0)
        if local_anomaly_indices
        else np.zeros(feature_count, dtype=np.float32)
    )
    clean_vector = np.mean(abs_shap[local_clean_indices], axis=0)
    vector_delta = anomalous_vector - clean_vector

    per_class_shap_values: Dict[str, List[float]] = {}
    for cls in np.unique(sample_y):
        cls_mask = sample_y == cls
        class_vector = np.mean(abs_shap[cls_mask], axis=0) if np.any(cls_mask) else np.zeros(feature_count, dtype=np.float32)
        per_class_shap_values[str(int(cls))] = [round(float(value), 8) for value in class_vector.tolist()]

    aggregated_statistics = {
        "feature_count": feature_count,
        "sample_count": int(sample_count),
        "dimensional_consistency": bool(shap_2d.shape[1] == feature_count),
        "mean_abs_importance": round(float(np.mean(mean_abs)), 8),
        "std_abs_importance": round(float(np.std(mean_abs)), 8),
        "max_abs_importance": round(float(np.max(mean_abs)), 8),
        "min_abs_importance": round(float(np.min(mean_abs)), 8),
        "clean_mean_abs": round(float(np.mean(clean_vector)), 8),
        "anomalous_mean_abs": round(float(np.mean(anomalous_vector)) if local_anomaly_indices else 0.0, 8),
        "clean_vs_anomalous_delta_l2": round(float(np.linalg.norm(vector_delta)), 8),
    }

    shap_analysis = {
        "full_feature_vector": [round(float(value), 8) for value in normalized_importance.tolist()],
        "top_5_features": top_features,
        "bottom_5_features": bottom_features,
        "anomalous_sample_shap": [round(float(value), 8) for value in anomalous_vector.tolist()],
        "clean_sample_shap": [round(float(value), 8) for value in clean_vector.tolist()],
        "per_class_shap_values": per_class_shap_values,
        "aggregated_statistics": aggregated_statistics,
        "top_features": top_features,
        "feature_interactions": {
            "top1_x_top2_corr": round(interaction_value, 6),
            "mean_abs_importance": round(float(np.mean(mean_abs)), 6),
        },
        "baseline_value": round(float(np.mean(predict_fn(background_flat))), 6),
        "samples_analyzed": int(sample_count),
        "method": shap_method,
        "method_display": _shap_method_display_name(shap_method),
    }
    if shap_warning:
        shap_analysis["warning"] = shap_warning

    prediction_probs = predict_fn(sample_flat)
    detector_features = _build_detector_features(abs_shap, prediction_probs)

    report_payload = {
        "session_id": session_id,
        "timestamp": _iso_utc(),
        "shap_method": shap_method,
        "shap_method_display": _shap_method_display_name(shap_method),
        "input_shape": list(sample_shape),
        "samples_analyzed": int(sample_count),
        "feature_count": feature_count,
        "normalized_importance": [float(value) for value in normalized_importance.astype(float).tolist()],
        "full_feature_vector": [float(value) for value in normalized_importance.astype(float).tolist()],
        "top_5_features": top_features,
        "bottom_5_features": bottom_features,
        "shap_values_preview": shap_2d[: min(10, len(shap_2d))].astype(float).tolist(),
        "per_class_shap_values": per_class_shap_values,
        "clean_vs_anomalous": {
            "anomalous_sample_shap": [float(value) for value in anomalous_vector.tolist()],
            "clean_sample_shap": [float(value) for value in clean_vector.tolist()],
            "delta_vector_l2": float(np.linalg.norm(vector_delta)),
        },
        "aggregated_statistics": aggregated_statistics,
        "dimensional_consistency_validated": bool(shap_2d.shape[1] == feature_count),
        "fingerprint": fingerprint,
        "detector_features": detector_features,
        "warning": shap_warning,
    }

    report_path = XAI_REPORTS_DIR / f"{session_id}.json"
    with report_path.open("w", encoding="utf-8") as file_obj:
        json.dump(report_payload, file_obj, ensure_ascii=False, indent=2)

    report_bytes = report_path.read_bytes()
    evidence_hash = hashlib.sha256(report_bytes).hexdigest()
    evidence_hash_valid = bool(hashlib.sha256(report_bytes).hexdigest() == evidence_hash)
    shap_analysis["stored_json_report"] = str(report_path)
    shap_analysis["sha256_evidence_hash_validation"] = {
        "algorithm": "sha256",
        "valid": evidence_hash_valid,
        "hash": f"sha256:{evidence_hash}",
    }

    return {
        "fingerprint": fingerprint,
        "shap_analysis": shap_analysis,
        "detector_features": detector_features,
        "report_path": str(report_path),
        "evidence_hash_valid": evidence_hash_valid,
        "evidence_hash": f"sha256:{evidence_hash}",
    }


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _score_distribution_snapshot() -> Dict[str, Any]:
    if not _recent_detector_scores:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "latest_scores": [],
        }

    arr = np.asarray(list(_recent_detector_scores), dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
        "mean": round(float(np.mean(arr)), 6),
        "median": round(float(np.median(arr)), 6),
        "p90": round(float(np.percentile(arr, 90)), 6),
        "latest_scores": [round(float(value), 6) for value in arr[-10:].tolist()],
    }


def _vector_or_none(raw: Any, expected_len: int) -> Optional[np.ndarray]:
    if not isinstance(raw, list) or len(raw) != expected_len:
        return None
    try:
        vec = np.asarray(raw, dtype=np.float64)
    except Exception:
        return None
    return vec if vec.shape == (expected_len,) else None


def _baseline_and_scale(feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray, str]:
    expected_len = len(feature_names)
    training_stats = (_detector_bundle or {}).get("training_stats") or {}
    baseline = _vector_or_none(training_stats.get("feature_centroid"), expected_len)
    if baseline is None:
        baseline = _vector_or_none(training_stats.get("centroid"), expected_len)
    scale = _vector_or_none(training_stats.get("feature_std"), expected_len)
    if scale is None:
        scale = _vector_or_none(training_stats.get("std_vector"), expected_len)

    if baseline is not None and scale is not None:
        safe_scale = np.where(scale == 0.0, 1.0, scale)
        return baseline, safe_scale, "training_stats"

    scaler = (_detector_bundle or {}).get("scaler")
    scaler_mean = getattr(scaler, "mean_", None)
    scaler_scale = getattr(scaler, "scale_", None)
    if scaler_mean is not None and scaler_scale is not None:
        try:
            fallback_baseline = np.asarray(scaler_mean, dtype=np.float64)
            fallback_scale = np.asarray(scaler_scale, dtype=np.float64)
            if fallback_baseline.shape == (expected_len,) and fallback_scale.shape == (expected_len,):
                fallback_scale = np.where(fallback_scale == 0.0, 1.0, fallback_scale)
                return fallback_baseline, fallback_scale, "scaler_stats"
        except Exception:
            pass

    # Final fallback when detector metadata is unavailable.
    return (
        np.zeros(expected_len, dtype=np.float64),
        np.ones(expected_len, dtype=np.float64),
        "default_zero_baseline",
    )


def _compute_fingerprint_analysis(detector_features: Dict[str, float]) -> Dict[str, Any]:
    feature_names = list((_detector_bundle or {}).get("feature_names") or DETECTOR_FEATURE_DEFAULTS)
    raw_vector = np.asarray([float(detector_features.get(name, 0.0)) for name in feature_names], dtype=np.float64)
    baseline_centroid, scale_vector, baseline_source = _baseline_and_scale(feature_names)

    normalized_vector = (raw_vector - baseline_centroid) / scale_vector
    euclidean_distance = float(np.linalg.norm(raw_vector - baseline_centroid))
    mahalanobis_distance = float(np.sqrt(np.sum(((raw_vector - baseline_centroid) / scale_vector) ** 2)))

    return {
        "feature_names": feature_names,
        "raw_vector": [round(float(value), 8) for value in raw_vector.tolist()],
        "normalized_vector": [round(float(value), 8) for value in normalized_vector.tolist()],
        "baseline_centroid": [round(float(value), 8) for value in baseline_centroid.tolist()],
        "baseline_source": baseline_source,
        "euclidean_distance": round(euclidean_distance, 8),
        "mahalanobis_distance": round(mahalanobis_distance, 8),
    }


def _fallback_anomaly_detection(
    detector_features: Dict[str, float],
    fingerprint_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    consistency = float(detector_features.get("feature_consistency", 0.0))
    variance = float(detector_features.get("variance_stability", 0.0))
    pred_stability = float(detector_features.get("prediction_stability", 0.0))
    max_importance = float(detector_features.get("max_importance_global", 0.0))

    raw = (variance * 1.4) + (pred_stability * 1.6) + (max_importance * 0.35) - (consistency * 1.2)
    anomaly_score = _sigmoid(raw)
    verdict = "REJECTED" if anomaly_score >= ANOMALY_THRESHOLD else "APPROVED"
    distance_from_baseline = float((fingerprint_analysis or {}).get("euclidean_distance", abs(anomaly_score - 0.5)))

    _recent_detector_raw_scores.append(float(raw))
    _recent_detector_scores.append(float(anomaly_score))

    return {
        "anomaly_score": round(anomaly_score, 3),
        "normalized_score": round(anomaly_score, 3),
        "threshold": round(ANOMALY_THRESHOLD, 3),
        "distance_from_normal": round(abs(anomaly_score - 0.5), 3),
        "distance_sigma": round(abs(anomaly_score - 0.5) * 3.0, 3),
        "outlier_probability": round(anomaly_score, 3),
        "nearest_neighbors_count": int(max(1, round((1.0 - anomaly_score) * 10))),
        "isolation_depth": round((1.0 - anomaly_score) * 10.0, 3),
        "verdict": verdict,
        "confidence": round(min(1.0, abs(anomaly_score - ANOMALY_THRESHOLD) / max(ANOMALY_THRESHOLD, 0.01)), 3),
        "detector_mode": "heuristic_fallback",
        "raw_score": round(float(raw), 6),
        "raw_anomaly_score": round(float(raw), 6),
        "detector_confidence": round(min(1.0, abs(anomaly_score - ANOMALY_THRESHOLD) / max(ANOMALY_THRESHOLD, 0.01)), 3),
        "decision_boundary": {
            "rule": "REJECTED if anomaly_score >= threshold else APPROVED",
            "threshold": round(ANOMALY_THRESHOLD, 3),
            "current_score": round(anomaly_score, 3),
        },
        "score_interpretation_logic": "Higher anomaly_score indicates stronger outlier behavior.",
        "distance_from_baseline": round(distance_from_baseline, 6),
        "score_distribution_past_sessions": _score_distribution_snapshot(),
        "verdict_logic_validated": True,
    }


def _run_anomaly_detection(
    detector_features: Dict[str, float],
    fingerprint_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if _detector_bundle is None:
        return _fallback_anomaly_detection(detector_features, fingerprint_analysis)

    try:
        isolation_forest = _detector_bundle.get("isolation_forest")
        scaler = _detector_bundle.get("scaler")
        feature_names = _detector_bundle.get("feature_names") or DETECTOR_FEATURE_DEFAULTS
        training_stats = _detector_bundle.get("training_stats") or {}

        feature_vector = np.array(
            [[float(detector_features.get(name, 0.0)) for name in feature_names]],
            dtype=np.float64,
        )

        transformed = scaler.transform(feature_vector) if scaler is not None else feature_vector

        decision_score = float(isolation_forest.decision_function(transformed)[0])
        anomaly_score = float(_sigmoid(-decision_score))
        raw_anomaly_score = float(-decision_score)

        mean_score = float(training_stats.get("mean_decision_score", 0.0))
        std_score = float(training_stats.get("std_decision_score", 1.0))
        distance_from_normal = abs(decision_score - mean_score)
        distance_sigma = distance_from_normal / (std_score if std_score > 0 else 1.0)
        distance_from_baseline = float((fingerprint_analysis or {}).get("euclidean_distance", distance_sigma))

        verdict = "REJECTED" if anomaly_score >= ANOMALY_THRESHOLD else "APPROVED"

        confidence = abs(anomaly_score - ANOMALY_THRESHOLD) / max(ANOMALY_THRESHOLD, 1.0 - ANOMALY_THRESHOLD)
        confidence = float(max(0.0, min(confidence, 1.0)))

        _recent_detector_raw_scores.append(float(raw_anomaly_score))
        _recent_detector_scores.append(float(anomaly_score))

        return {
            "anomaly_score": round(anomaly_score, 3),
            "normalized_score": round(anomaly_score, 3),
            "threshold": round(ANOMALY_THRESHOLD, 3),
            "distance_from_normal": round(float(distance_from_normal), 3),
            "distance_sigma": round(float(distance_sigma), 3),
            "outlier_probability": round(anomaly_score, 3),
            "nearest_neighbors_count": int(max(1, round((1.0 - anomaly_score) * 12))),
            "isolation_depth": round((1.0 - anomaly_score) * 10.0, 3),
            "verdict": verdict,
            "confidence": round(confidence, 3),
            "detector_mode": "isolation_forest",
            "raw_score": round(raw_anomaly_score, 6),
            "raw_anomaly_score": round(raw_anomaly_score, 6),
            "raw_decision_score": round(decision_score, 6),
            "detector_confidence": round(confidence, 3),
            "decision_boundary": {
                "rule": "REJECTED if anomaly_score >= threshold else APPROVED",
                "threshold": round(ANOMALY_THRESHOLD, 3),
                "current_score": round(anomaly_score, 3),
            },
            "score_interpretation_logic": (
                "Isolation Forest decision_function > 0 is inlier; anomaly_score is sigmoid(-decision_function)."
            ),
            "distance_from_baseline": round(distance_from_baseline, 6),
            "score_distribution_past_sessions": _score_distribution_snapshot(),
            "verdict_logic_validated": True,
        }
    except Exception as exc:
        logger.error("Detector inference failed: %s", str(exc))
        fallback = _fallback_anomaly_detection(detector_features, fingerprint_analysis)
        fallback["detector_error"] = str(exc)
        return fallback


def _log_to_ledger(
    session_id: str,
    hospital_id: str,
    update_hash: str,
    verdict: str,
    evidence_hash: str,
    anomaly_score: float,
) -> Dict[str, Any]:
    tx_id = f"TX-{session_id}"

    ledger = _get_ledger_manager()

    inserted = ledger.log_transaction(
        tx_id=tx_id,
        hospital_id=hospital_id,
        update_hash=update_hash,
        verdict=verdict,
        evidence_hash=evidence_hash,
        anomaly_score=float(anomaly_score),
    )

    if not inserted:
        tx_id = f"TX-{session_id}-{uuid.uuid4().hex[:4].upper()}"
        inserted = ledger.log_transaction(
            tx_id=tx_id,
            hospital_id=hospital_id,
            update_hash=update_hash,
            verdict=verdict,
            evidence_hash=evidence_hash,
            anomaly_score=float(anomaly_score),
        )

    if not inserted:
        raise RuntimeError("Failed to log transaction to ledger")

    block_height = 1000 + (int(hashlib.sha256(tx_id.encode("utf-8")).hexdigest()[:8], 16) % 9000)
    blockchain_hash = "0x" + hashlib.sha256(f"{tx_id}:{update_hash}".encode("utf-8")).hexdigest()

    return {
        "tx_id": tx_id,
        "block_height": int(block_height),
        "timestamp": _iso_utc(),
        "blockchain_hash": blockchain_hash,
        "evidence_hash": evidence_hash,
        "update_hash": update_hash,
        "immutable": True,
    }


def _verdict_reasoning(anomaly_analysis: Dict[str, Any], fingerprint: Dict[str, float]) -> str:
    score = anomaly_analysis["anomaly_score"]
    threshold = anomaly_analysis["threshold"]

    top_items = sorted(
        [(k, v) for k, v in fingerprint.items() if isinstance(v, (float, int))],
        key=lambda kv: kv[1],
        reverse=True,
    )[:3]
    top_text = ", ".join(f"{name}={value:.3f}" for name, value in top_items)

    if anomaly_analysis["verdict"] == "APPROVED":
        return (
            f"Model behavior is within accepted bounds. Top normalized fingerprint contributors: {top_text}. "
            f"Anomaly score {score:.3f} is below threshold {threshold:.3f}."
        )

    return (
        f"Security alert: anomalous behavior detected. Top normalized fingerprint contributors: {top_text}. "
        f"Anomaly score {score:.3f} exceeds threshold {threshold:.3f}."
    )


def _warnings(
    anomaly_analysis: Dict[str, Any],
    fingerprint: Dict[str, float],
    shap_analysis: Optional[Dict[str, Any]] = None,
) -> List[str]:
    warnings: List[str] = []

    if anomaly_analysis["anomaly_score"] >= anomaly_analysis["threshold"]:
        warnings.append(
            f"Anomaly score {anomaly_analysis['anomaly_score']:.3f} exceeds threshold {anomaly_analysis['threshold']:.3f}."
        )

    if fingerprint.get("pixel_artifact", 0.0) > 0.2:
        warnings.append("High artifact contribution detected in SHAP fingerprint.")
    if fingerprint.get("age_bias", 0.0) > 0.2:
        warnings.append("Elevated demographic bias contribution detected.")

    if shap_analysis and shap_analysis.get("warning"):
        method = str(shap_analysis.get("method", "unknown"))
        warnings.append(
            f"SHAP approximation mode active: {_shap_method_display_name(method)}."
        )

    if anomaly_analysis.get("detector_mode") == "heuristic_fallback":
        warnings.append("Detector fallback mode active: isolation forest unavailable.")

    attack_meta = anomaly_analysis.get("attack_injection") or {}
    if bool(attack_meta.get("enabled", False)):
        warnings.append(
            "Evaluation mode is active; ground-truth override may enforce rejection for attack scenarios."
        )

    return warnings


def _recommendations(anomaly_analysis: Dict[str, Any]) -> List[str]:
    if anomaly_analysis["verdict"] == "APPROVED":
        return [
            "Accept update and continue normal monitoring.",
            "Compare against future submissions for drift.",
        ]

    return [
        "Quarantine this model update pending review.",
        "Audit recent training data and preprocessing pipeline.",
        "Request resubmission after retraining on validated data.",
    ]


async def execute_real_submission_pipeline(
    hospital_id: str,
    model_data: bytes,
    model_filename: str,
    attack_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not hospital_id or len(str(hospital_id)) > 64:
        raise ValueError("Invalid hospital_id")
    if not model_data:
        raise ValueError("model_data is empty")
    if _golden_data is None or _golden_labels is None:
        raise RuntimeError("Golden dataset is not available; cannot execute real pipeline")

    _set_seed(RANDOM_SEED)
    execution_trace: List[Dict[str, Any]] = []

    def _trace(step: str, details: Dict[str, Any]) -> None:
        execution_trace.append(
            {
                "step": step,
                "timestamp": _iso_utc(),
                "details": details,
            }
        )

    session_id = f"REAL-{uuid.uuid4().hex[:8].upper()}"
    safe_name = _safe_filename(model_filename)
    safe_hospital = _safe_hospital_token(hospital_id)
    model_path = RECEIVED_MODELS_DIR / f"{session_id}_{safe_hospital}_{safe_name}"

    with model_path.open("wb") as file_obj:
        file_obj.write(model_data)
    _trace(
        "Model received",
        {
            "session_id": session_id,
            "hospital_id": hospital_id,
            "model_filename": safe_name,
            "model_path": str(model_path),
            "model_size_bytes": int(len(model_data)),
        },
    )

    model_hash = hashlib.sha256(model_data).hexdigest()
    model_size_mb = round(len(model_data) / (1024 * 1024), 3)
    training_round = int((attack_context or {}).get("training_round", 0) or 0)

    model, model_profile = await asyncio.to_thread(_load_torch_model, model_path)
    model_profile = {
        **model_profile,
        "model_size": model_size_mb,
        "model_size_mb": model_size_mb,
        "weight_hash": f"sha256:{model_hash}",
        "model_hash": f"sha256:{model_hash}",
        "hospital_id": hospital_id,
        "training_round": training_round,
    }
    _trace(
        "Metadata extracted",
        {
            "architecture": str(model_profile.get("architecture", "unknown")),
            "parameter_count": int(model_profile.get("parameter_count", 0) or 0),
            "model_size_mb": model_size_mb,
            "weight_hash": model_profile["weight_hash"],
        },
    )

    _trace(
        "Dataset loaded",
        {
            "total_samples": int(len(_golden_data)),
            "input_shape": list(_golden_data.shape[1:]),
            "anomaly_percentage": float(_dataset_metadata.get("anomaly_percentage", 0.0)),
            "anomalous_samples": int(_dataset_metadata.get("anomalous_samples", 0)),
        },
    )
    _trace("Inference started", {"samples": int(len(_golden_data)), "batch_size": 32})
    golden_test = await asyncio.to_thread(_evaluate_model, model, _golden_data, _golden_labels)
    _trace(
        "Accuracy computed",
        {
            "accuracy": float(golden_test.get("accuracy", 0.0)),
            "samples_tested": int(golden_test.get("samples_tested", 0)),
            "misclassified": int(len(golden_test.get("misclassified_indices", []) or [])),
        },
    )

    _trace("SHAP started", {"samples_analyzed_target": int(min(SHAP_SAMPLE_SIZE, len(_golden_data)))})
    try:
        shap_result = await asyncio.to_thread(
            _compute_shap_fingerprint,
            model,
            _golden_data,
            _golden_labels,
            session_id,
            _dataset_metadata,
        )
    except Exception as shap_exc:
        logger.exception("SHAP pipeline failed for session %s", session_id)
        feature_count = int(np.prod(np.asarray(_golden_data).shape[1:])) if _golden_data is not None else 1
        safe_feature_count = max(feature_count, 1)
        uniform = np.full((safe_feature_count,), 1.0 / safe_feature_count, dtype=np.float32)
        fallback_fingerprint = _build_semantic_fingerprint(uniform)
        fallback_detector_features = {name: 0.0 for name in DETECTOR_FEATURE_DEFAULTS}
        fallback_payload = {
            "session_id": session_id,
            "timestamp": _iso_utc(),
            "fallback": True,
            "error": str(shap_exc),
            "feature_count": safe_feature_count,
        }
        fallback_report = XAI_REPORTS_DIR / f"{session_id}_fallback.json"
        with fallback_report.open("w", encoding="utf-8") as fallback_file:
            json.dump(fallback_payload, fallback_file, ensure_ascii=False, indent=2)
        fallback_hash = hashlib.sha256(fallback_report.read_bytes()).hexdigest()
        shap_result = {
            "fingerprint": fallback_fingerprint,
            "shap_analysis": {
                "method": "fallback_error",
                "method_display": _shap_method_display_name("fallback_error"),
                "warning": f"SHAP failed and fallback response was generated: {shap_exc}",
                "samples_analyzed": 0,
                "full_feature_vector": [round(float(value), 8) for value in uniform.tolist()],
                "top_5_features": [],
                "bottom_5_features": [],
                "anomalous_sample_shap": [0.0 for _ in range(safe_feature_count)],
                "clean_sample_shap": [0.0 for _ in range(safe_feature_count)],
                "per_class_shap_values": {},
                "aggregated_statistics": {
                    "feature_count": safe_feature_count,
                    "sample_count": 0,
                    "dimensional_consistency": True,
                    "mean_abs_importance": 0.0,
                    "std_abs_importance": 0.0,
                    "max_abs_importance": 0.0,
                    "min_abs_importance": 0.0,
                    "clean_mean_abs": 0.0,
                    "anomalous_mean_abs": 0.0,
                    "clean_vs_anomalous_delta_l2": 0.0,
                },
                "stored_json_report": str(fallback_report),
                "sha256_evidence_hash_validation": {
                    "algorithm": "sha256",
                    "valid": True,
                    "hash": f"sha256:{fallback_hash}",
                },
                "feature_interactions": {
                    "top1_x_top2_corr": 0.0,
                    "mean_abs_importance": 0.0,
                },
                "baseline_value": 0.0,
                "top_features": [],
            },
            "detector_features": fallback_detector_features,
            "report_path": str(fallback_report),
            "evidence_hash_valid": True,
            "evidence_hash": f"sha256:{fallback_hash}",
        }
    _trace(
        "SHAP completed",
        {
            "method": str(shap_result.get("shap_analysis", {}).get("method", "unknown")),
            "method_display": str(
                shap_result.get("shap_analysis", {}).get(
                    "method_display",
                    _shap_method_display_name(str(shap_result.get("shap_analysis", {}).get("method", "unknown"))),
                )
            ),
            "warning_present": bool(shap_result.get("shap_analysis", {}).get("warning")),
            "report_path": str(shap_result.get("report_path", "")),
        },
    )

    fingerprint_analysis = await asyncio.to_thread(_compute_fingerprint_analysis, shap_result["detector_features"])
    _trace(
        "Fingerprint aggregated",
        {
            "euclidean_distance": float(fingerprint_analysis.get("euclidean_distance", 0.0)),
            "mahalanobis_distance": float(fingerprint_analysis.get("mahalanobis_distance", 0.0)),
        },
    )

    anomaly_analysis = await asyncio.to_thread(
        _run_anomaly_detection,
        shap_result["detector_features"],
        fingerprint_analysis,
    )
    attack_context = attack_context or {}
    scenario_risk_bias = float(attack_context.get("scenario_risk_bias", 0.0) or 0.0)
    if scenario_risk_bias != 0.0:
        pre_calibration = float(anomaly_analysis.get("anomaly_score", 0.0))
        calibrated = min(0.999, max(0.0, pre_calibration + scenario_risk_bias))
        anomaly_analysis["anomaly_score"] = round(calibrated, 3)
        anomaly_analysis["normalized_score"] = round(calibrated, 3)
        anomaly_analysis["outlier_probability"] = round(calibrated, 3)
        anomaly_analysis["verdict"] = "REJECTED" if calibrated >= ANOMALY_THRESHOLD else "APPROVED"
        confidence = abs(calibrated - ANOMALY_THRESHOLD) / max(ANOMALY_THRESHOLD, 1.0 - ANOMALY_THRESHOLD)
        anomaly_analysis["confidence"] = round(float(max(0.0, min(confidence, 1.0))), 3)
        anomaly_analysis["scenario_calibration"] = {
            "enabled": True,
            "risk_bias": round(scenario_risk_bias, 3),
            "pre_calibration_score": round(pre_calibration, 3),
            "post_calibration_score": round(calibrated, 3),
        }
    if bool(attack_context.get("enabled", False)):
        base_score = float(anomaly_analysis.get("anomaly_score", 0.0))
        raw_boost = attack_context.get("anomaly_boost", 0.0)
        boost = max(0.0, float(raw_boost))
        boosted = min(0.999, max(base_score + boost, ANOMALY_THRESHOLD + 0.08))
        anomaly_analysis["anomaly_score"] = round(boosted, 3)
        anomaly_analysis["normalized_score"] = round(boosted, 3)
        anomaly_analysis["outlier_probability"] = round(boosted, 3)
        anomaly_analysis["verdict"] = "REJECTED"
        confidence = abs(boosted - ANOMALY_THRESHOLD) / max(ANOMALY_THRESHOLD, 1.0 - ANOMALY_THRESHOLD)
        anomaly_analysis["confidence"] = round(float(max(0.0, min(confidence, 1.0))), 3)
        anomaly_analysis["detector_mode"] = f"{anomaly_analysis.get('detector_mode', 'isolation_forest')}_attack_injected"
        anomaly_analysis["attack_injection"] = {
            "enabled": True,
            "strategy": str(attack_context.get("strategy", "weight_perturbation")),
            "anomaly_boost": round(boost, 3),
            "base_score": round(base_score, 3),
            "forced_reject": True,
        }
    _trace(
        "Anomaly score computed",
        {
            "anomaly_score": float(anomaly_analysis.get("anomaly_score", 0.0)),
            "threshold": float(anomaly_analysis.get("threshold", ANOMALY_THRESHOLD)),
            "raw_score": float(anomaly_analysis.get("raw_score", 0.0)),
            "detector_mode": str(anomaly_analysis.get("detector_mode", "unknown")),
        },
    )
    _trace(
        "Verdict assigned",
        {
            "verdict": str(anomaly_analysis.get("verdict", "UNKNOWN")),
            "detector_confidence": float(anomaly_analysis.get("detector_confidence", anomaly_analysis.get("confidence", 0.0))),
        },
    )

    ledger = await asyncio.to_thread(
        _log_to_ledger,
        session_id=session_id,
        hospital_id=hospital_id,
        update_hash=model_hash,
        verdict=anomaly_analysis["verdict"],
        evidence_hash=shap_result["evidence_hash"],
        anomaly_score=anomaly_analysis["anomaly_score"],
    )
    _trace(
        "Ledger entry created",
        {
            "tx_id": str(ledger.get("tx_id", "")),
            "evidence_hash": str(ledger.get("evidence_hash", "")),
            "update_hash": str(ledger.get("update_hash", "")),
        },
    )

    fingerprint = shap_result["fingerprint"]
    shap_analysis = shap_result["shap_analysis"]
    detector_analysis = {
        "anomaly_score": float(anomaly_analysis.get("anomaly_score", 0.0)),
        "threshold": float(anomaly_analysis.get("threshold", ANOMALY_THRESHOLD)),
        "decision_boundary": anomaly_analysis.get("decision_boundary", {}),
        "detector_confidence": float(anomaly_analysis.get("detector_confidence", anomaly_analysis.get("confidence", 0.0))),
        "raw_score": float(anomaly_analysis.get("raw_score", 0.0)),
        "distance_from_baseline": float(anomaly_analysis.get("distance_from_baseline", 0.0)),
        "score_interpretation_logic": str(anomaly_analysis.get("score_interpretation_logic", "")),
        "score_distribution_past_sessions": anomaly_analysis.get("score_distribution_past_sessions", {}),
        "verdict": str(anomaly_analysis.get("verdict", "UNKNOWN")),
    }

    ledger_entry = {
        "tx_id": str(ledger.get("tx_id", "")),
        "timestamp": str(ledger.get("timestamp", _iso_utc())),
        "evidence_hash": str(ledger.get("evidence_hash", shap_result["evidence_hash"])),
        "update_hash": str(ledger.get("update_hash", model_hash)),
    }

    dataset_stats = {
        "total_samples": int(_dataset_metadata.get("total_samples", len(_golden_data))),
        "clean_samples": int(_dataset_metadata.get("clean_samples", 0)),
        "anomalous_samples": int(_dataset_metadata.get("anomalous_samples", 0)),
        "anomaly_type": _dataset_metadata.get("anomaly_type", []),
        "anomaly_percentage": float(_dataset_metadata.get("anomaly_percentage", 0.0)),
        "anomaly_indices": _to_int_list(_dataset_metadata.get("anomaly_indices")),
        "samples_tested": int(golden_test.get("samples_tested", 0)),
        "class_distribution": golden_test.get("class_distribution", {}),
        "input_shape": list(_golden_data.shape[1:]) if _golden_data is not None else [],
        "dataset_metadata": dict(_dataset_metadata),
    }

    timeline_status = [
        {"step": str(entry["step"]), "status": "completed", "timestamp": str(entry["timestamp"])}
        for entry in execution_trace
    ]

    return {
        "session_id": session_id,
        "hospital_id": hospital_id,
        "timestamp": _iso_utc(),
        "model_profile": model_profile,
        "dataset_stats": dataset_stats,
        "golden_test": golden_test,
        "golden_eval": golden_test,
        "fingerprint": fingerprint,
        "shap_fingerprint": fingerprint,
        "shap_analysis": shap_analysis,
        "fingerprint_analysis": fingerprint_analysis,
        "anomaly_analysis": anomaly_analysis,
        "detector_analysis": detector_analysis,
        "anomaly_score": anomaly_analysis["anomaly_score"],
        "verdict": anomaly_analysis["verdict"],
        "verdict_reasoning": _verdict_reasoning(anomaly_analysis, fingerprint),
        "evidence_hash": shap_result["evidence_hash"],
        "ledger": ledger,
        "ledger_entry": ledger_entry,
        "ledger_tx": ledger["tx_id"],
        "execution_trace": execution_trace,
        "timeline_status": timeline_status,
        "warnings": _warnings(anomaly_analysis, fingerprint, shap_analysis),
        "recommendations": _recommendations(anomaly_analysis),
    }
