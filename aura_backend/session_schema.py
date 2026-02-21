"""Helpers to normalize session payloads across demo and real modes."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List


def _derive_dataset_stats(payload: Dict[str, Any]) -> Dict[str, Any]:
    golden = payload.get("golden_test", {}) or {}
    class_distribution = golden.get("class_distribution", {}) or {}
    samples_tested = int(golden.get("samples_tested", 0) or 0)

    if not class_distribution and isinstance(golden.get("confusion_matrix"), list):
        matrix = golden.get("confusion_matrix") or []
        if len(matrix) >= 2 and all(isinstance(row, list) and len(row) >= 2 for row in matrix[:2]):
            negatives = int(matrix[0][0]) + int(matrix[0][1])
            positives = int(matrix[1][0]) + int(matrix[1][1])
            class_distribution = {"negative": negatives, "positive": positives}

    return {
        "samples_tested": samples_tested,
        "class_distribution": class_distribution,
        "input_shape": payload.get("dataset_stats", {}).get("input_shape", []),
    }


def _default_timeline(ts: str) -> List[Dict[str, Any]]:
    labels = [
        "model_received",
        "metadata_extracted",
        "golden_evaluation",
        "shap_fingerprint",
        "anomaly_detection",
        "ledger_logged",
        "response_ready",
    ]
    return [{"step": label, "status": "completed", "timestamp": ts} for label in labels]


def normalize_session_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    now_ts = payload.get("timestamp") or datetime.utcnow().isoformat() + "Z"

    ledger = payload.get("ledger") or {}
    anomaly = payload.get("anomaly_analysis") or {}
    golden = payload.get("golden_test") or {}
    fingerprint = payload.get("fingerprint") or {}

    payload["timestamp"] = now_ts
    payload["golden_test"] = golden
    payload["fingerprint"] = fingerprint
    payload["anomaly_analysis"] = anomaly
    payload["ledger"] = ledger

    anomaly_defaults = {
        "anomaly_score": float(anomaly.get("anomaly_score", 0.0) or 0.0),
        "normalized_score": float(anomaly.get("normalized_score", anomaly.get("anomaly_score", 0.0)) or 0.0),
        "threshold": float(anomaly.get("threshold", 0.72) or 0.72),
        "distance_from_normal": float(anomaly.get("distance_from_normal", 0.0) or 0.0),
        "distance_sigma": float(anomaly.get("distance_sigma", 0.0) or 0.0),
        "outlier_probability": float(anomaly.get("outlier_probability", anomaly.get("anomaly_score", 0.0)) or 0.0),
        "nearest_neighbors_count": int(anomaly.get("nearest_neighbors_count", 0) or 0),
        "isolation_depth": float(anomaly.get("isolation_depth", 0.0) or 0.0),
        "confidence": float(anomaly.get("confidence", 0.0) or 0.0),
        "detector_mode": str(anomaly.get("detector_mode", "isolation_forest")),
        "verdict": str(anomaly.get("verdict", payload.get("verdict", "UNKNOWN"))),
        "attack_injection": anomaly.get("attack_injection"),
        "scenario_calibration": anomaly.get("scenario_calibration"),
    }
    payload["anomaly_analysis"] = anomaly_defaults

    shap_analysis = payload.get("shap_analysis") or {}
    if "method" not in shap_analysis:
        shap_analysis["method"] = "not_available"
    payload["shap_analysis"] = shap_analysis

    payload["dataset_stats"] = payload.get("dataset_stats") or _derive_dataset_stats(payload)
    payload["golden_eval"] = payload.get("golden_eval") or golden
    payload["shap_fingerprint"] = payload.get("shap_fingerprint") or fingerprint
    payload["anomaly_score"] = float(payload.get("anomaly_score", anomaly_defaults["anomaly_score"]) or 0.0)

    evidence_hash = payload.get("evidence_hash") or ledger.get("evidence_hash")
    if evidence_hash is None:
        evidence_hash = ""
    payload["evidence_hash"] = evidence_hash

    ledger_tx = payload.get("ledger_tx") or ledger.get("tx_id")
    if ledger_tx is None:
        ledger_tx = ""
    payload["ledger_tx"] = ledger_tx

    timeline = payload.get("timeline_status")
    if not isinstance(timeline, list) or not timeline:
        payload["timeline_status"] = _default_timeline(now_ts)

    return payload
