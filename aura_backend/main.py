import logging
import io
import base64
import hashlib
import os
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional at runtime
    Image = None

try:
    from config import AURA_MODE, LEDGER_DB_PATH
except ImportError:  # pragma: no cover - import-path fallback
    from aura_backend.config import AURA_MODE, LEDGER_DB_PATH  # type: ignore
try:
    from ledger_manager import LedgerManager
    from pipeline_runner import PipelineRunManager
    from real_handler import (
        check_components_status,
        execute_real_submission_pipeline,
        get_dataset_inspection_payload,
        initialize_real_mode_components,
    )
    from session_schema import normalize_session_payload
    from session_storage import SessionStorage
except ImportError:  # pragma: no cover - import-path fallback
    from aura_backend.ledger_manager import LedgerManager  # type: ignore
    from aura_backend.pipeline_runner import PipelineRunManager  # type: ignore
    from aura_backend.real_handler import (  # type: ignore
        check_components_status,
        execute_real_submission_pipeline,
        get_dataset_inspection_payload,
        initialize_real_mode_components,
    )
    from aura_backend.session_schema import normalize_session_payload  # type: ignore
    from aura_backend.session_storage import SessionStorage  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("AURA_CORS_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]
ALLOWED_HOSPITAL_IDS = {f"HOSPITAL{i}" for i in range(1, 11)} | {f"HOSP{i}" for i in range(1, 11)}
AURA_API_KEY = os.getenv("AURA_API_KEY", "").strip()
AUTH_EXEMPT_PATHS = {"/", "/system/mode"}
AUTH_EXEMPT_PREFIXES = ("/docs", "/redoc", "/openapi.json")

app = FastAPI(
    title="AURA - Federated Learning Security System",
    version="2.1.0",
    description="Real-time backend for federated model security analysis.",
)
# Note: This prototype supports optional API-key protection via AURA_API_KEY.
# Production deployments should enforce OAuth2/JWT per site.

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ledger_manager = LedgerManager(LEDGER_DB_PATH)
session_storage = SessionStorage(max_sessions=250)
pipeline_run_manager = PipelineRunManager(session_storage=session_storage)
startup_time = datetime.now(timezone.utc)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _error_payload(code: str, message: str, path: str, status_code: int) -> Dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
            "status_code": int(status_code),
            "path": path,
            "timestamp": _iso_now(),
        },
    }


def _is_auth_exempt(path: str) -> bool:
    if path in AUTH_EXEMPT_PATHS:
        return True
    return any(path.startswith(prefix) for prefix in AUTH_EXEMPT_PREFIXES)


def _extract_api_key(request: Request) -> str:
    direct = str(request.headers.get("x-api-key", "")).strip()
    if direct:
        return direct
    auth = str(request.headers.get("authorization", "")).strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


@app.middleware("http")
async def api_key_guard(request: Request, call_next):
    if (
        not AURA_API_KEY
        or request.method.upper() == "OPTIONS"
        or _is_auth_exempt(request.url.path)
    ):
        return await call_next(request)

    provided = _extract_api_key(request)
    if provided != AURA_API_KEY:
        return JSONResponse(
            status_code=401,
            content=_error_payload(
                code="UNAUTHORIZED",
                message="Missing or invalid API key",
                path=str(request.url.path),
                status_code=401,
            ),
        )
    return await call_next(request)


def _to_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        cleaned = value.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned)
    except Exception:
        return None


def _hospital_numeric_id(value: Any) -> int:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return int(digits) if digits else 0


def _normalize_hospital_id(value: Any) -> str:
    token = str(value or "").strip().upper()
    if token.startswith("HOSPITAL") and token[8:].isdigit():
        return f"HOSPITAL{int(token[8:])}"
    if token.startswith("HOSP") and token[4:].isdigit():
        return f"HOSP{int(token[4:])}"
    digits = "".join(ch for ch in token if ch.isdigit())
    if digits:
        return f"HOSP{int(digits)}"
    return token


def _validate_hospital_id(value: Any) -> str:
    normalized = _normalize_hospital_id(value)
    if not normalized:
        raise HTTPException(status_code=400, detail="hospital_id is required")
    if normalized not in ALLOWED_HOSPITAL_IDS:
        allowed = ", ".join(f"HOSP{i}" for i in range(1, 11))
        raise HTTPException(
            status_code=422,
            detail=f"Invalid hospital_id '{value}'. Allowed values: {allowed}",
        )
    return normalized


def _find_session_by_tx_id(tx_id: str) -> Optional[Dict[str, Any]]:
    for session in _all_sessions():
        ledger = session.get("ledger") if isinstance(session.get("ledger"), dict) else {}
        ledger_entry = (
            session.get("ledger_entry")
            if isinstance(session.get("ledger_entry"), dict)
            else {}
        )
        ledger_tx = str(session.get("ledger_tx", "") or "")
        if (
            str(ledger.get("tx_id", "") or "") == tx_id
            or str(ledger_entry.get("tx_id", "") or "") == tx_id
            or ledger_tx == tx_id
        ):
            return session
    return None


def _resolve_evidence_report_path(session: Dict[str, Any]) -> Optional[Path]:
    shap_analysis = (
        session.get("shap_analysis")
        if isinstance(session.get("shap_analysis"), dict)
        else {}
    )
    candidate = shap_analysis.get("stored_json_report")
    if candidate:
        path = Path(str(candidate))
        if path.exists():
            return path
    return None


def _parse_hospital_numeric(value: Any) -> int:
    token = _normalize_hospital_id(value)
    digits = "".join(ch for ch in token if ch.isdigit())
    return int(digits) if digits else 0


def _encode_image_base64(sample: np.ndarray) -> str:
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


def _dataset_for_hospital(hospital_id: str, sample_preview_limit: int = 12) -> Dict[str, Any]:
    numeric_hospital = _parse_hospital_numeric(hospital_id)
    if numeric_hospital not in {1, 2, 3}:
        raise ValueError("Dataset explorer currently supports Hospital 1, 2, and 3.")

    scenario = pipeline_run_manager._scenario_for_hospital(hospital_id, False)  # pylint: disable=protected-access
    sample_offsets = scenario.get("sample_offsets") or {}
    sample_offset = int(sample_offsets.get(numeric_hospital, 0))
    noise_std = float(scenario.get("augmentation_noise_std", 0.0))

    loader, info = pipeline_run_manager._load_hospital_dataset(  # pylint: disable=protected-access
        numeric_hospital,
        256,
        sample_offset,
        noise_std,
        42 + numeric_hospital * 100,
    )
    tensor_x, tensor_y = loader.dataset.tensors  # type: ignore[attr-defined]
    data = tensor_x.detach().cpu().numpy()
    labels = tensor_y.detach().cpu().numpy()

    preview_count = max(1, min(int(sample_preview_limit), len(data)))
    images = [
        {
            "index": int(idx),
            "label": int(labels[idx]),
            "image_base64": _encode_image_base64(data[idx]),
        }
        for idx in range(preview_count)
    ]

    class_distribution = {
        str(int(label)): int(count)
        for label, count in zip(*np.unique(labels, return_counts=True))
    }
    represented_classes = sorted(int(k) for k in class_distribution.keys())
    start_idx = int(info.get("data_slice_start", 0))
    end_idx = int(start_idx + info.get("samples", preview_count) - 1)

    golden_payload = get_dataset_inspection_payload(sample_limit=100)
    preprocessing = list(scenario.get("preprocessing_pipeline", []))

    return {
        "hospital_id": f"HOSP{numeric_hospital}",
        "display_name": f"Hospital {numeric_hospital}",
        "dataset_slice": {
            "sample_count": int(info.get("samples", len(data))),
            "index_range": {"start": start_idx, "end": end_idx},
            "noise_level": round(noise_std, 4),
            "classes_represented": represented_classes,
            "input_shape": info.get("input_shape", []),
            "dataset_variant": scenario.get("dataset_variant", ""),
        },
        "class_distribution": class_distribution,
        "sample_images": images,
        "golden_validation_set": {
            "sample_count": int(golden_payload.get("total_samples", 0)),
            "class_distribution": golden_payload.get("class_distribution", {}),
            "sample_images": golden_payload.get("sample_preview", []),
            "anomaly_percentage": float(golden_payload.get("anomaly_percentage", 0.0)),
        },
        "preprocessing_steps": preprocessing,
        "scenario": {
            "scenario_id": scenario.get("scenario_id"),
            "dataset_name": scenario.get("dataset_name"),
            "dataset_variant": scenario.get("dataset_variant"),
            "participant_hospitals": scenario.get("participant_hospitals", []),
        },
    }


def _build_fingerprint_comparison(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    submitted = session.get("shap_fingerprint") or session.get("fingerprint") or {}
    if not isinstance(submitted, dict):
        submitted = {}
    named_features = [
        key
        for key in submitted.keys()
        if isinstance(submitted.get(key), (float, int)) and str(key) != "age"
    ]

    baseline_centroid = (
        (session.get("fingerprint_analysis") or {}).get("baseline_centroid")
        if isinstance(session.get("fingerprint_analysis"), dict)
        else []
    )
    baseline_vector = baseline_centroid if isinstance(baseline_centroid, list) else []

    comparison: List[Dict[str, Any]] = []
    for idx, feature in enumerate(named_features):
        submitted_value = float(submitted.get(feature, 0.0) or 0.0)
        baseline_value = float(baseline_vector[idx]) if idx < len(baseline_vector) else 0.0
        comparison.append(
            {
                "feature": str(feature),
                "baseline": round(baseline_value, 6),
                "submitted": round(submitted_value, 6),
                "delta": round(submitted_value - baseline_value, 6),
            }
        )
    return sorted(comparison, key=lambda row: abs(float(row["delta"])), reverse=True)


def _build_evidence_payload(session: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_session_payload(dict(session))
    anomaly = normalized.get("anomaly_analysis") if isinstance(normalized.get("anomaly_analysis"), dict) else {}
    detector = normalized.get("detector_analysis") if isinstance(normalized.get("detector_analysis"), dict) else {}
    ledger = normalized.get("ledger_entry") if isinstance(normalized.get("ledger_entry"), dict) else {}
    warnings = normalized.get("warnings") if isinstance(normalized.get("warnings"), list) else []
    recommendations = normalized.get("recommendations") if isinstance(normalized.get("recommendations"), list) else []
    shap_analysis = normalized.get("shap_analysis") if isinstance(normalized.get("shap_analysis"), dict) else {}

    anomaly_inspection = get_dataset_inspection_payload(sample_limit=24)
    fingerprint_comparison = _build_fingerprint_comparison(normalized)
    score = float(anomaly.get("anomaly_score", detector.get("anomaly_score", 0.0)) or 0.0)
    threshold = float(anomaly.get("threshold", detector.get("threshold", 0.72)) or 0.72)
    score_distribution = anomaly.get("score_distribution_past_sessions") if isinstance(anomaly.get("score_distribution_past_sessions"), dict) else {}
    above_threshold = int(score_distribution.get("above_threshold", 0) or 0)
    total = int(score_distribution.get("total_scores", 0) or 0)
    percentile = round((above_threshold / max(total, 1)) * 100.0, 2)

    return {
        "session_id": normalized.get("session_id", ""),
        "timestamp": normalized.get("timestamp", _iso_now()),
        "hospital_id": normalized.get("hospital_id", ""),
        "verdict": normalized.get("verdict", "UNKNOWN"),
        "ledger_tx_id": str(ledger.get("tx_id", normalized.get("ledger_tx", ""))),
        "accusation": (
            f"On {normalized.get('timestamp', _iso_now())}, {normalized.get('hospital_id', '')} submitted model "
            f"{normalized.get('session_id', '')}. The AURA Sentinel flagged this submission as "
            f"{normalized.get('verdict', 'UNKNOWN')}."
        ),
        "dataset_contamination_evidence": {
            "total_samples_tested": int((normalized.get("golden_eval") or {}).get("samples_tested", 0) or 0),
            "anomalies_detected": int(anomaly_inspection.get("anomaly_count", 0)),
            "anomaly_percentage": float(anomaly_inspection.get("anomaly_percentage", 0.0)),
            "anomaly_types": anomaly_inspection.get("anomaly_type", []),
            "sample_preview": anomaly_inspection.get("sample_preview", []),
        },
        "behavioral_fingerprint_comparison": fingerprint_comparison,
        "anomaly_diagnosis": {
            "score": round(score, 6),
            "threshold": round(threshold, 6),
            "score_exceeds_threshold": bool(score >= threshold),
            "anomalous_percentile_estimate": percentile,
            "detector_mode": anomaly.get("detector_mode", detector.get("detector_mode", "unknown")),
        },
        "immutable_record": {
            "tx_id": str(ledger.get("tx_id", "")),
            "evidence_hash": str(ledger.get("evidence_hash", normalized.get("evidence_hash", ""))),
            "update_hash": str(ledger.get("update_hash", "")),
            "timestamp": str(ledger.get("timestamp", normalized.get("timestamp", _iso_now()))),
        },
        "shap_named_features": shap_analysis.get("top_5_features") or shap_analysis.get("top_features") or [],
        "warnings": [str(item) for item in warnings],
        "recommendations": [str(item) for item in recommendations],
    }


def _all_sessions() -> List[Dict[str, Any]]:
    return session_storage.all_sessions()


def _recent_sessions(limit: int = 20) -> List[Dict[str, Any]]:
    return session_storage.list_full_sessions(limit=limit)


def _ledger_connected() -> bool:
    return ledger_manager.is_connected()


def _dashboard_stats() -> Dict[str, Any]:
    sessions = _all_sessions()
    total = len(sessions)
    verdicts: Dict[str, int] = defaultdict(int)
    for session in sessions:
        verdicts[str(session.get("verdict", "UNKNOWN"))] += 1

    approved = verdicts.get("APPROVED", 0)
    rejected = verdicts.get("REJECTED", 0)
    processing = verdicts.get("PROCESSING", 0)
    approval_rate = (approved / total * 100.0) if total > 0 else 0.0
    security_effectiveness = (rejected / total * 100.0) if total > 0 else 0.0
    uptime_seconds = int((datetime.now(timezone.utc) - startup_time).total_seconds())

    return {
        "total_submissions": total,
        "approved": approved,
        "rejected": rejected,
        "active_sessions": min(total, 25),
        "processing": processing,
        "approval_rate": round(approval_rate, 2),
        "security_effectiveness": round(security_effectiveness, 2),
        "uptime": f"{uptime_seconds}s",
    }


def _hydrate_sessions_from_ledger(limit: int = 100) -> int:
    if not ledger_manager.is_connected():
        return 0
    hydrated = 0
    rows = ledger_manager.get_recent_transactions(limit=limit)
    for tx in rows:
        tx_id = str(tx.get("tx_id", "") or "")
        if not tx_id.startswith("TX-"):
            continue
        session_id = tx_id[3:]
        if session_storage.get_session(session_id) is not None:
            continue
        payload = {
            "session_id": session_id,
            "hospital_id": tx.get("hospital_id", ""),
            "timestamp": tx.get("timestamp", _iso_now()),
            "verdict": tx.get("verdict", "UNKNOWN"),
            "anomaly_score": float(tx.get("anomaly_score", 0.0) or 0.0),
            "anomaly_analysis": {
                "anomaly_score": float(tx.get("anomaly_score", 0.0) or 0.0),
                "threshold": 0.72,
                "verdict": tx.get("verdict", "UNKNOWN"),
            },
            "ledger_entry": {
                "tx_id": tx_id,
                "timestamp": tx.get("timestamp", _iso_now()),
                "evidence_hash": tx.get("evidence_hash", ""),
                "update_hash": tx.get("update_hash", ""),
            },
            "ledger": {
                "tx_id": tx_id,
                "timestamp": tx.get("timestamp", _iso_now()),
                "evidence_hash": tx.get("evidence_hash", ""),
                "update_hash": tx.get("update_hash", ""),
            },
            "ledger_tx": tx_id,
            "warnings": [],
            "recommendations": [],
        }
        session_storage.store_session(session_id, normalize_session_payload(payload))
        hydrated += 1
    return hydrated


def _daily_metrics(days: int) -> List[Dict[str, Any]]:
    safe_days = max(1, min(int(days), 365))
    sessions = _all_sessions()

    day_map: Dict[str, Dict[str, int]] = {}
    for idx in range(safe_days):
        day = (datetime.now(timezone.utc) - timedelta(days=(safe_days - idx - 1))).date().isoformat()
        day_map[day] = {"submissions": 0, "approved": 0, "rejected": 0}

    for session in sessions:
        ts = _to_dt(session.get("timestamp"))
        if ts is None:
            continue
        day_key = ts.date().isoformat()
        if day_key not in day_map:
            continue
        day_map[day_key]["submissions"] += 1
        if session.get("verdict") == "APPROVED":
            day_map[day_key]["approved"] += 1
        if session.get("verdict") == "REJECTED":
            day_map[day_key]["rejected"] += 1

    return [{"date": day, **values} for day, values in day_map.items()]


def _attack_metrics() -> List[Dict[str, Any]]:
    counts = {
        "DATA_POISONING": 0,
        "MODEL_POISONING": 0,
        "INFERENCE_ATTACK": 0,
        "BYZANTINE": 0,
    }

    for session in _all_sessions():
        if session.get("verdict") != "REJECTED":
            continue
        warnings = " ".join(session.get("warnings", [])).lower()
        score = float((session.get("anomaly_analysis") or {}).get("anomaly_score", 0.0) or 0.0)
        if "artifact" in warnings:
            counts["DATA_POISONING"] += 1
        elif "bias" in warnings:
            counts["INFERENCE_ATTACK"] += 1
        elif score >= 0.9:
            counts["MODEL_POISONING"] += 1
        else:
            counts["BYZANTINE"] += 1

    return [{"type": key, "count": value} for key, value in counts.items()]


def _hospital_performance() -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for session in _all_sessions():
        hid = _hospital_numeric_id(session.get("hospital_id"))
        if hid <= 0:
            continue
        grouped[hid].append(session)

    results: List[Dict[str, Any]] = []
    for hid, rows in grouped.items():
        submissions = len(rows)
        approved = sum(1 for row in rows if row.get("verdict") == "APPROVED")
        acc_values = [
            float((row.get("golden_test") or {}).get("accuracy", 0.0) or 0.0)
            for row in rows
        ]
        avg_acc = sum(acc_values) / len(acc_values) if acc_values else 0.0
        approval_rate = (approved / submissions * 100.0) if submissions else 0.0
        results.append(
            {
                "hospital_id": hid,
                "hospital_name": f"Hospital {hid}",
                "submissions": submissions,
                "approval_rate": round(approval_rate, 2),
                "avg_accuracy": round(avg_acc, 4),
            }
        )
    return sorted(results, key=lambda row: row["approval_rate"], reverse=True)


def _security_threats(status: Optional[str]) -> List[Dict[str, Any]]:
    safe_status = (status or "ALL").upper()
    threats: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc)

    for session in _recent_sessions(limit=250):
        if session.get("verdict") != "REJECTED":
            continue

        ts = _to_dt(session.get("timestamp")) or now
        age_hours = (now - ts).total_seconds() / 3600.0
        score = float((session.get("anomaly_analysis") or {}).get("anomaly_score", 0.0) or 0.0)
        warnings = " ".join(session.get("warnings", [])).lower()
        if "artifact" in warnings:
            threat_type = "DATA_POISONING"
        elif "bias" in warnings:
            threat_type = "INFERENCE_ATTACK"
        elif score >= 0.9:
            threat_type = "MODEL_POISONING"
        else:
            threat_type = "BYZANTINE"

        if score >= 0.9:
            severity = "CRITICAL"
        elif score >= 0.8:
            severity = "HIGH"
        elif score >= 0.72:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        if age_hours <= 24:
            threat_status = "ACTIVE"
        elif age_hours <= 72:
            threat_status = "INVESTIGATING"
        else:
            threat_status = "RESOLVED"

        threat = {
            "id": f"threat_{session.get('session_id')}",
            "type": threat_type,
            "severity": severity,
            "hospital_id": _hospital_numeric_id(session.get("hospital_id")),
            "description": (session.get("verdict_reasoning") or "Rejected model submission")[:240],
            "detected_at": session.get("timestamp") or _iso_now(),
            "status": threat_status,
        }
        if safe_status == "ALL" or threat_status == safe_status:
            threats.append(threat)

    return threats[:100]


def _security_recommendations() -> List[Dict[str, Any]]:
    threats = _security_threats(status="ALL")
    active = [t for t in threats if t["status"] == "ACTIVE"]
    critical = [t for t in threats if t["severity"] == "CRITICAL"]

    recommendations = []
    if critical:
        recommendations.append(
            {
                "id": "rec_critical_quarantine",
                "priority": "HIGH",
                "title": "Quarantine Critical Updates",
                "description": f"{len(critical)} critical submissions require immediate containment.",
                "action_required": "Block deployment for affected sessions and start forensic review.",
            }
        )
    if active:
        recommendations.append(
            {
                "id": "rec_active_monitor",
                "priority": "MEDIUM",
                "title": "Increase Active Monitoring",
                "description": f"{len(active)} active threats are still being triaged.",
                "action_required": "Increase review frequency and validate next hospital submissions.",
            }
        )
    if not recommendations:
        recommendations.append(
            {
                "id": "rec_baseline",
                "priority": "LOW",
                "title": "Maintain Baseline Monitoring",
                "description": "No critical threats detected in recent sessions.",
                "action_required": "Continue routine checks and retrain detector periodically.",
            }
        )
    return recommendations


def _latest_sentinel_log() -> Optional[Dict[str, Any]]:
    latest = _recent_sessions(limit=1)
    if not latest:
        return None
    session = latest[0]
    shap_analysis = session.get("shap_analysis") or {}
    warning = str(shap_analysis.get("warning", "") or "")
    shap_status = "FAILED" if warning and "failed" in warning.lower() else "COMPLETED"
    if not shap_analysis:
        shap_status = "PROCESSING"

    steps = []
    for idx, step in enumerate(session.get("timeline_status", []) or []):
        steps.append(
            {
                "id": str(step.get("step") or f"step_{idx}"),
                "label": str(step.get("step") or f"step_{idx}"),
                "completed": step.get("status", "completed") == "completed",
                "timestamp": step.get("timestamp"),
            }
        )

    return {
        "model_id": session.get("session_id"),
        "hospital_id": _hospital_numeric_id(session.get("hospital_id")),
        "golden_set_accuracy": float((session.get("golden_test") or {}).get("accuracy", 0.0) or 0.0),
        "shap_status": shap_status,
        "isolation_score": float((session.get("anomaly_analysis") or {}).get("anomaly_score", 0.0) or 0.0),
        "verdict": session.get("verdict"),
        "ledger_tx_id": (session.get("ledger") or {}).get("tx_id", ""),
        "timestamp": session.get("timestamp") or _iso_now(),
        "processing_steps": steps,
    }


class SystemModeResponse(BaseModel):
    mode: str


class SystemHealthResponse(BaseModel):
    mode: str
    detector_loaded: bool
    ledger_connected: bool
    golden_set_loaded: bool


class PipelineStartRequest(BaseModel):
    hospital_id: str = "HOSP1"
    rounds: int = 2
    local_epochs: int = 1
    max_samples_per_hospital: int = 256
    attack_mode: bool = False


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    code = f"HTTP_{int(exc.status_code)}"
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_payload(code=code, message=detail, path=str(request.url.path), status_code=exc.status_code),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled backend exception on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content=_error_payload(
            code="INTERNAL_ERROR",
            message=f"Unexpected backend error: {exc}",
            path=str(request.url.path),
            status_code=500,
        ),
    )


@app.get("/system/mode", response_model=SystemModeResponse)
async def get_system_mode() -> SystemModeResponse:
    return SystemModeResponse(mode=AURA_MODE)


@app.get("/system/health", response_model=SystemHealthResponse)
async def get_system_health() -> SystemHealthResponse:
    status = check_components_status()
    return SystemHealthResponse(
        mode=AURA_MODE,
        detector_loaded=bool(status["detector_loaded"]),
        ledger_connected=bool(status["ledger_connected"]),
        golden_set_loaded=bool(status["golden_set_loaded"]),
    )


@app.get("/status")
async def get_status() -> Dict[str, Any]:
    status = check_components_status()
    recent = _recent_sessions(limit=1)
    last_run = recent[0].get("timestamp") if recent else None
    return {
        "backend": "online",
        "model_loaded": bool(status["detector_loaded"]),
        "last_run": last_run,
    }


@app.post("/sentinel/submit_update")
async def submit_update(
    hospital_id: str = Form(...),
    model_file: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    start_time = time.perf_counter()
    safe_hospital_id = _validate_hospital_id(hospital_id)

    try:
        if model_file is None:
            raise HTTPException(status_code=400, detail="model_file is required")

        model_data = await model_file.read()
        if not model_data:
            raise HTTPException(status_code=400, detail="Uploaded model_file is empty")

        payload = await execute_real_submission_pipeline(
            hospital_id=safe_hospital_id,
            model_data=model_data,
            model_filename=model_file.filename or "uploaded_model.pt",
        )

        payload = normalize_session_payload(payload)
        payload["processing_time"] = round(time.perf_counter() - start_time, 3)
        session_storage.store_session(payload["session_id"], payload)

        logger.info(
            "submission_complete mode=%s session=%s hospital=%s verdict=%s",
            AURA_MODE,
            payload.get("session_id"),
            safe_hospital_id,
            payload.get("verdict"),
        )
        return payload

    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning("submit_update validation failure for hospital=%s: %s", safe_hospital_id, str(exc))
        raise HTTPException(status_code=400, detail=f"Submission validation failed: {exc}") from exc
    except RuntimeError as exc:
        logger.error("submit_update runtime failure for hospital=%s: %s", safe_hospital_id, str(exc))
        raise HTTPException(status_code=503, detail=f"Submission unavailable: {exc}") from exc
    except Exception as exc:
        logger.exception("submit_update failed for hospital=%s", safe_hospital_id)
        raise HTTPException(status_code=500, detail=f"Submission failed: {exc}") from exc


@app.get("/session/{session_id}")
async def get_session(session_id: str) -> Dict[str, Any]:
    session = session_storage.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return normalize_session_payload(session)


@app.get("/sessions/list")
async def list_sessions(limit: int = 10) -> Dict[str, Any]:
    sessions = session_storage.list_sessions(limit=limit)
    return {
        "sessions": sessions,
        "count": len(sessions),
        "timestamp": _iso_now(),
    }


@app.get("/sentinel/report/{submission_id}")
async def get_sentinel_report(submission_id: str) -> Dict[str, Any]:
    session = session_storage.get_session(submission_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {submission_id} not found")
    return normalize_session_payload(session)


@app.get("/sentinel/reports")
async def get_sentinel_reports(limit: int = 20) -> Dict[str, Any]:
    sessions = [normalize_session_payload(item) for item in _recent_sessions(limit=limit)]
    return {
        "reports": sessions,
        "count": len(sessions),
        "timestamp": _iso_now(),
    }


@app.get("/sentinel/detection_stats")
async def get_sentinel_detection_stats() -> Dict[str, Any]:
    return _dashboard_stats()


@app.get("/dataset/inspection")
async def get_dataset_inspection(sample_limit: int = 12) -> Dict[str, Any]:
    safe_limit = max(2, min(int(sample_limit), 40))
    try:
        return get_dataset_inspection_payload(sample_limit=safe_limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"Dataset inspection unavailable: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid dataset inspection request: {exc}") from exc


@app.get("/api/dataset/hospital/{hospital_id}")
async def get_hospital_dataset_view(hospital_id: str, sample_limit: int = 12) -> Dict[str, Any]:
    safe_hospital_id = _validate_hospital_id(hospital_id)
    try:
        return _dataset_for_hospital(
            hospital_id=safe_hospital_id,
            sample_preview_limit=max(4, min(int(sample_limit), 40)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/evidence/{session_id}")
async def get_evidence_by_session(session_id: str) -> Dict[str, Any]:
    session = session_storage.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    try:
        return _build_evidence_payload(session)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build evidence payload: {exc}") from exc


@app.get("/api/evidence/tx/{tx_id}")
async def get_evidence_by_transaction(tx_id: str) -> Dict[str, Any]:
    session = _find_session_by_tx_id(tx_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"No session linked to transaction {tx_id}")
    try:
        return _build_evidence_payload(session)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build evidence payload: {exc}") from exc


@app.get("/sentinel/logs/latest")
async def get_sentinel_logs_latest() -> Dict[str, Any]:
    latest = _latest_sentinel_log()
    if latest is None:
        return {
            "model_id": "",
            "hospital_id": 0,
            "golden_set_accuracy": 0.0,
            "shap_status": "PROCESSING",
            "isolation_score": 0.0,
            "verdict": "PROCESSING",
            "ledger_tx_id": "",
            "timestamp": _iso_now(),
            "processing_steps": [],
        }
    return latest


@app.post("/sentinel/pipeline/start")
async def start_sentinel_pipeline(request: PipelineStartRequest) -> Dict[str, Any]:
    hospital_id = _validate_hospital_id(request.hospital_id)

    try:
        run = await pipeline_run_manager.start_run(
            hospital_id=hospital_id,
            rounds=request.rounds,
            local_epochs=request.local_epochs,
            max_samples_per_hospital=request.max_samples_per_hospital,
            attack_mode=bool(request.attack_mode),
        )
        return run
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to start pipeline run for hospital=%s", hospital_id)
        raise HTTPException(status_code=500, detail=f"Failed to start pipeline: {exc}") from exc


@app.get("/sentinel/pipeline/status/{run_id}")
async def get_sentinel_pipeline_status(run_id: str) -> Dict[str, Any]:
    run = await pipeline_run_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Pipeline run {run_id} not found")
    return run


@app.get("/sentinel/pipeline/latest")
async def get_sentinel_pipeline_latest() -> Dict[str, Any]:
    run = await pipeline_run_manager.get_latest_run()
    return {"run": run}


@app.get("/sentinel/pipeline/download/{run_id}")
async def download_sentinel_pipeline_model(run_id: str) -> FileResponse:
    model_path = await pipeline_run_manager.get_generated_model_path(run_id)
    if model_path is None:
        raise HTTPException(status_code=404, detail=f"No generated model artifact for run {run_id}")
    return FileResponse(
        path=model_path,
        media_type="application/octet-stream",
        filename=model_path.name,
    )


@app.get("/ledger/transactions")
async def get_ledger_transactions(limit: int = 20) -> Dict[str, Any]:
    if not _ledger_connected():
        raise HTTPException(status_code=503, detail="Ledger database unavailable")
    rows = ledger_manager.get_recent_transactions(limit=limit)
    return {
        "transactions": rows,
        "count": len(rows),
        "timestamp": _iso_now(),
    }


@app.get("/ledger/transaction/{tx_id}")
async def get_ledger_transaction(tx_id: str) -> Dict[str, Any]:
    if not _ledger_connected():
        raise HTTPException(status_code=503, detail="Ledger database unavailable")
    tx = ledger_manager.get_transaction(tx_id)
    if tx is None:
        raise HTTPException(status_code=404, detail=f"Transaction {tx_id} not found")
    return tx


@app.get("/ledger/stats")
async def get_ledger_stats() -> Dict[str, Any]:
    if not _ledger_connected():
        raise HTTPException(status_code=503, detail="Ledger database unavailable")
    stats = ledger_manager.get_statistics()
    total = int(stats.get("total_transactions", 0))
    rejected = int(stats.get("rejected", 0))
    security_rate = (rejected / total * 100.0) if total > 0 else 0.0

    return {
        "total_transactions": total,
        "approved": int(stats.get("approved", 0)),
        "rejected": rejected,
        "security_rate": round(security_rate, 2),
        "blockchain_info": {
            "total_blocks": 1000 + total,
            "latest_hash": hashlib.sha256(f"{total}:{rejected}".encode("utf-8")).hexdigest(),
            "pending_transactions": 0,
            "last_updated": _iso_now(),
        },
    }


@app.get("/ledger/verify/{tx_id}")
async def verify_ledger_transaction(tx_id: str) -> Dict[str, Any]:
    if not _ledger_connected():
        raise HTTPException(status_code=503, detail="Ledger database unavailable")
    tx = ledger_manager.get_transaction(tx_id)
    if tx is None:
        return {
            "verified": False,
            "valid": False,
            "stored_hash": "",
            "computed_hash": "",
            "message": f"Transaction {tx_id} not found",
        }

    stored_hash = str(tx.get("evidence_hash", "") or "")
    session = _find_session_by_tx_id(tx_id)
    report_path = _resolve_evidence_report_path(session or {})
    if report_path is None:
        return {
            "verified": False,
            "valid": False,
            "stored_hash": stored_hash,
            "computed_hash": "",
            "message": f"Transaction {tx_id} found, but local evidence report is unavailable",
        }

    computed_hash = "sha256:" + hashlib.sha256(report_path.read_bytes()).hexdigest()
    valid = bool(stored_hash == computed_hash)
    return {
        "verified": valid,
        "valid": valid,
        "stored_hash": stored_hash,
        "computed_hash": computed_hash,
        "message": f"Transaction {tx_id} {'verified' if valid else 'hash mismatch'}",
    }


@app.get("/analytics/daily")
async def get_analytics_daily(days: int = 30) -> List[Dict[str, Any]]:
    return _daily_metrics(days=days)


@app.get("/analytics/attacks")
async def get_analytics_attacks() -> List[Dict[str, Any]]:
    return _attack_metrics()


@app.get("/analytics/hospitals")
async def get_analytics_hospitals() -> List[Dict[str, Any]]:
    return _hospital_performance()


@app.get("/security/threats")
async def get_security_threats(status: str = "ALL") -> List[Dict[str, Any]]:
    return _security_threats(status=status)


@app.get("/security/recommendations")
async def get_security_recommendations() -> List[Dict[str, Any]]:
    return _security_recommendations()


@app.get("/stats/summary")
async def get_stats_summary() -> Dict[str, Any]:
    return {
        "mode": AURA_MODE,
        "session_stats": session_storage.get_statistics(),
        "ledger_stats": ledger_manager.get_statistics(),
        "timestamp": _iso_now(),
    }


@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "service": "AURA Backend",
        "mode": AURA_MODE,
        "status": "ok",
        "endpoints": [
            "/system/mode",
            "/system/health",
            "/status",
            "/sentinel/submit_update",
            "/sentinel/pipeline/start",
            "/sentinel/pipeline/status/{run_id}",
            "/session/{session_id}",
            "/sentinel/reports",
            "/sentinel/detection_stats",
            "/dataset/inspection",
            "/api/dataset/hospital/{hospital_id}",
            "/api/evidence/{session_id}",
            "/api/evidence/tx/{tx_id}",
            "/sentinel/logs/latest",
            "/ledger/transactions",
            "/ledger/stats",
            "/analytics/daily",
            "/security/threats",
        ],
    }


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("AURA backend starting in %s mode", AURA_MODE)
    if not ledger_manager.initialize_database():
        logger.error("Ledger initialization failed at startup: %s", ledger_manager.last_error)
    try:
        initialize_real_mode_components()
    except Exception as exc:
        logger.error("Real mode component initialization failed: %s", str(exc))
    hydrated = _hydrate_sessions_from_ledger(limit=150)
    if hydrated > 0:
        logger.info("Hydrated %s sessions from ledger at startup", hydrated)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    ledger_manager.close()
    logger.info("AURA backend stopped")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
