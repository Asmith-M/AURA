from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime

class ModelSubmissionRequest(BaseModel):
    """Request model for submitting a model update"""
    hospital_id: int = Field(..., description="ID of the hospital submitting the model")
    model_weights: List[List[float]] = Field(..., description="Model weights as nested lists")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(default_factory=datetime.now)

class ModelSubmissionResponse(BaseModel):
    """Response model for model submission"""
    submission_id: str
    hospital_id: int
    status: str  # "RECEIVED", "PROCESSING", "PROCESSED", "REJECTED"
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)

class ModelInferenceRequest(BaseModel):
    """Request model for running inference on golden set"""
    model_weights: List[List[float]]
    golden_set_indices: Optional[List[int]] = None

class ModelInferenceResponse(BaseModel):
    """Response model for inference results"""
    predictions: List[float]
    confidence_scores: List[float]
    test_accuracy: float
    raw_outputs: List[List[float]]

class BehavioralReportRequest(BaseModel):
    """Request for behavioral analysis"""
    model_weights: List[List[float]]
    golden_set_indices: Optional[List[int]] = None

class BehavioralReportResponse(BaseModel):
    """Response for behavioral analysis"""
    submission_id: str
    hospital_id: int
    behavioral_fingerprint: Dict[str, float]
    test_accuracy: float
    raw_predictions: List[List[float]]
    timestamp: datetime = Field(default_factory=datetime.now)

class StatusResponse(BaseModel):
    """Response for system status"""
    status: str
    uptime: str
    active_models: int
    processed_updates: int
    pending_analyses: int