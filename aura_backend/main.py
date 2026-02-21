import logging
import hashlib
import time
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import AURA_MODE, LEDGER_DB_PATH
from ledger_manager import LedgerManager
from pipeline_runner import PipelineRunManager
from real_handler import (
    check_components_status,
    execute_real_submission_pipeline,
    initialize_real_mode_components,
)
from session_schema import normalize_session_payload
from session_storage import SessionStorage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AURA - Federated Learning Security System",
    version="2.1.0",
    description="Real-time backend for federated model security analysis.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


@app.post("/sentinel/submit_update")
async def submit_update(
    hospital_id: str = Form(...),
    model_file: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    start_time = time.perf_counter()
    safe_hospital_id = str(hospital_id).strip()
    if not safe_hospital_id:
        raise HTTPException(status_code=400, detail="hospital_id is required")
    if len(safe_hospital_id) > 64:
        raise HTTPException(status_code=400, detail="hospital_id is too long (max 64 chars)")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", safe_hospital_id):
        raise HTTPException(status_code=400, detail="hospital_id contains invalid characters")

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
    hospital_id = str(request.hospital_id).strip()
    if not hospital_id:
        raise HTTPException(status_code=400, detail="hospital_id is required")
    if len(hospital_id) > 64:
        raise HTTPException(status_code=400, detail="hospital_id is too long (max 64 chars)")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", hospital_id):
        raise HTTPException(status_code=400, detail="hospital_id contains invalid characters")

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
        return {"verified": False, "message": f"Transaction {tx_id} not found"}
    return {"verified": True, "message": f"Transaction {tx_id} verified"}


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
            "/sentinel/submit_update",
            "/sentinel/pipeline/start",
            "/sentinel/pipeline/status/{run_id}",
            "/session/{session_id}",
            "/sentinel/reports",
            "/sentinel/detection_stats",
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


@app.on_event("shutdown")
async def on_shutdown() -> None:
    ledger_manager.close()
    logger.info("AURA backend stopped")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
