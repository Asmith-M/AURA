"""Helpers to normalize session payloads for full Sentinel transparency."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List


_REQUIRED_TRACE_STEPS = [
    "Model received",
    "Metadata extracted",
    "Dataset loaded",
    "Inference started",
    "Accuracy computed",
    "SHAP started",
    "SHAP completed",
    "Fingerprint aggregated",
    "Anomaly score computed",
    "Verdict assigned",
    "Ledger entry created",
]


def _derive_dataset_stats(payload: Dict[str, Any]) -> Dict[str, Any]:
    golden = payload.get("golden_eval") or payload.get("golden_test") or {}
    existing = payload.get("dataset_stats") or {}

    class_distribution = existing.get("class_distribution") or golden.get("class_distribution") or {}
    samples_tested = int(existing.get("samples_tested") or golden.get("samples_tested") or 0)

    return {
        "total_samples": int(existing.get("total_samples") or samples_tested),
        "clean_samples": int(existing.get("clean_samples") or max(samples_tested - int(existing.get("anomalous_samples", 0) or 0), 0)),
        "anomalous_samples": int(existing.get("anomalous_samples") or 0),
        "anomaly_type": existing.get("anomaly_type") or [],
        "anomaly_percentage": float(existing.get("anomaly_percentage") or 0.0),
        "anomaly_indices": existing.get("anomaly_indices") or [],
        "samples_tested": samples_tested,
        "class_distribution": class_distribution,
        "input_shape": existing.get("input_shape") or [],
        "dataset_metadata": existing.get("dataset_metadata") or {},
    }


def _default_execution_trace(ts: str) -> List[Dict[str, Any]]:
    return [
        {
            "step": step,
            "timestamp": ts,
            "details": {},
        }
        for step in _REQUIRED_TRACE_STEPS
    ]


def _default_timeline(trace: List[Dict[str, Any]], ts: str) -> List[Dict[str, Any]]:
    if not trace:
        return [{"step": step, "status": "completed", "timestamp": ts} for step in _REQUIRED_TRACE_STEPS]
    return [
        {
            "step": str(item.get("step", "unknown")),
            "status": "completed",
            "timestamp": item.get("timestamp", ts),
        }
        for item in trace
    ]


def normalize_session_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    now_ts = payload.get("timestamp") or datetime.utcnow().isoformat() + "Z"

    payload["timestamp"] = now_ts

    model_profile = payload.get("model_profile") or {}
    anomaly = payload.get("anomaly_analysis") or payload.get("detector_analysis") or {}
    golden = payload.get("golden_eval") or payload.get("golden_test") or {}
    shap_analysis = payload.get("shap_analysis") or {}
    fingerprint_analysis = payload.get("fingerprint_analysis") or {}
    ledger = payload.get("ledger") or {}
    ledger_entry = payload.get("ledger_entry") or {}

    payload["model_profile"] = {
        "architecture": model_profile.get("architecture", "unknown"),
        "parameter_count": int(model_profile.get("parameter_count", 0) or 0),
        "model_size": model_profile.get("model_size", model_profile.get("model_size_mb", 0.0)),
        "model_size_mb": float(model_profile.get("model_size_mb", model_profile.get("model_size", 0.0)) or 0.0),
        "weight_hash": model_profile.get("weight_hash", model_profile.get("model_hash", "")),
        "model_hash": model_profile.get("model_hash", model_profile.get("weight_hash", "")),
        "hospital_id": model_profile.get("hospital_id", payload.get("hospital_id", "")),
        "training_round": int(model_profile.get("training_round", 0) or 0),
        "parameters": model_profile.get("parameters", "0.00M"),
    }

    payload["dataset_stats"] = _derive_dataset_stats(payload)

    payload["golden_eval"] = {
        **golden,
        "accuracy": float(golden.get("accuracy", 0.0) or 0.0),
        "confusion_matrix": golden.get("confusion_matrix") or [],
        "precision_per_class": golden.get("precision_per_class") or golden.get("per_class_precision") or {},
        "recall_per_class": golden.get("recall_per_class") or golden.get("per_class_recall") or {},
        "f1_per_class": golden.get("f1_per_class") or {},
        "confidence_distribution": golden.get("confidence_distribution") or {},
        "misclassified_indices": golden.get("misclassified_indices") or [],
    }
    payload["golden_test"] = payload["golden_eval"]

    payload["shap_analysis"] = {
        "full_feature_vector": shap_analysis.get("full_feature_vector") or [],
        "top_5_features": shap_analysis.get("top_5_features") or shap_analysis.get("top_features") or [],
        "bottom_5_features": shap_analysis.get("bottom_5_features") or [],
        "anomalous_sample_shap": shap_analysis.get("anomalous_sample_shap") or [],
        "clean_sample_shap": shap_analysis.get("clean_sample_shap") or [],
        "per_class_shap_values": shap_analysis.get("per_class_shap_values") or {},
        "aggregated_statistics": shap_analysis.get("aggregated_statistics") or {},
        "method": shap_analysis.get("method", "not_available"),
        "warning": shap_analysis.get("warning"),
        "stored_json_report": shap_analysis.get("stored_json_report"),
        "sha256_evidence_hash_validation": shap_analysis.get("sha256_evidence_hash_validation") or {},
        "feature_interactions": shap_analysis.get("feature_interactions") or {},
        "top_features": shap_analysis.get("top_features") or shap_analysis.get("top_5_features") or [],
        "baseline_value": shap_analysis.get("baseline_value", 0.0),
        "samples_analyzed": int(shap_analysis.get("samples_analyzed", 0) or 0),
    }

    payload["fingerprint_analysis"] = {
        "raw_vector": fingerprint_analysis.get("raw_vector") or [],
        "normalized_vector": fingerprint_analysis.get("normalized_vector") or [],
        "baseline_centroid": fingerprint_analysis.get("baseline_centroid") or [],
        "euclidean_distance": float(fingerprint_analysis.get("euclidean_distance", 0.0) or 0.0),
        "mahalanobis_distance": float(fingerprint_analysis.get("mahalanobis_distance", 0.0) or 0.0),
        "feature_names": fingerprint_analysis.get("feature_names") or [],
    }

    detector_analysis = payload.get("detector_analysis") or {
        "anomaly_score": anomaly.get("anomaly_score", payload.get("anomaly_score", 0.0)),
        "threshold": anomaly.get("threshold", 0.72),
        "decision_boundary": anomaly.get("decision_boundary") or {},
        "detector_confidence": anomaly.get("detector_confidence", anomaly.get("confidence", 0.0)),
    }

    payload["detector_analysis"] = {
        **detector_analysis,
        "anomaly_score": float(detector_analysis.get("anomaly_score", 0.0) or 0.0),
        "threshold": float(detector_analysis.get("threshold", 0.72) or 0.72),
        "detector_confidence": float(detector_analysis.get("detector_confidence", 0.0) or 0.0),
        "decision_boundary": detector_analysis.get("decision_boundary") or {},
    }

    payload["anomaly_analysis"] = {
        **anomaly,
        "anomaly_score": float(anomaly.get("anomaly_score", payload["detector_analysis"]["anomaly_score"]) or 0.0),
        "normalized_score": float(anomaly.get("normalized_score", anomaly.get("anomaly_score", 0.0)) or 0.0),
        "threshold": float(anomaly.get("threshold", payload["detector_analysis"]["threshold"]) or 0.72),
        "detector_confidence": float(anomaly.get("detector_confidence", anomaly.get("confidence", 0.0)) or 0.0),
        "confidence": float(anomaly.get("confidence", anomaly.get("detector_confidence", 0.0)) or 0.0),
        "decision_boundary": anomaly.get("decision_boundary") or payload["detector_analysis"].get("decision_boundary") or {},
        "verdict": str(anomaly.get("verdict", payload.get("verdict", "UNKNOWN"))),
    }

    payload["anomaly_score"] = float(payload.get("anomaly_score", payload["anomaly_analysis"]["anomaly_score"]) or 0.0)

    entry = {
        "tx_id": ledger_entry.get("tx_id", ledger.get("tx_id", payload.get("ledger_tx", ""))),
        "timestamp": ledger_entry.get("timestamp", ledger.get("timestamp", now_ts)),
        "evidence_hash": ledger_entry.get("evidence_hash", ledger.get("evidence_hash", payload.get("evidence_hash", ""))),
        "update_hash": ledger_entry.get("update_hash", ledger.get("update_hash", "")),
    }
    payload["ledger_entry"] = entry
    payload["ledger"] = {**ledger, **entry}
    payload["ledger_tx"] = str(entry["tx_id"] or "")
    payload["evidence_hash"] = str(entry["evidence_hash"] or payload.get("evidence_hash", ""))

    trace = payload.get("execution_trace")
    if not isinstance(trace, list) or not trace:
        trace = _default_execution_trace(now_ts)
    payload["execution_trace"] = trace

    timeline = payload.get("timeline_status")
    if not isinstance(timeline, list) or not timeline:
        payload["timeline_status"] = _default_timeline(trace, now_ts)

    payload.setdefault("warnings", [])
    payload.setdefault("recommendations", [])

    return payload
