from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List
import uuid
import time
from datetime import datetime
import asyncio
import hashlib
import requests

from api.models import (
    ModelSubmissionRequest, ModelSubmissionResponse, 
    ModelInferenceRequest, ModelInferenceResponse,
    BehavioralReportRequest, BehavioralReportResponse,
    StatusResponse
)
from core.model_validator import model_validator
from core.shap_analyzer import shap_analyzer
from core.fingerprint import fingerprint_manager
from storage.model_storage import model_storage

# Import detector components
from detector.detector import detection_engine

# Import ledger integration
import httpx

# Create FastAPI app
app = FastAPI(
    title="Aura Sentinel API",
    description="XAI-based security sentinel for federated learning",
    version="0.4.0"  # Updated version
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state tracking
sentinel_state = {
    'uptime': time.time(),
    'active_models': 0,
    'processed_updates': 0,
    'pending_analyses': 0,
    'completed_analyses': 0,
    'detections_performed': 0,
    'approvals': 0,
    'rejections': 0,
    'ledger_interactions': 0
}

# Ledger API URL
LEDGER_API_URL = "http://localhost:8001"

@app.get("/", response_model=StatusResponse)
async def root():
    """Root endpoint for health check"""
    uptime_seconds = int(time.time() - sentinel_state['uptime'])
    uptime_str = f"{uptime_seconds // 3600}h {(uptime_seconds % 3600) // 60}m {uptime_seconds % 60}s"
    
    return StatusResponse(
        status="running",
        uptime=uptime_str,
        active_models=sentinel_state['active_models'],
        processed_updates=sentinel_state['processed_updates'],
        pending_analyses=sentinel_state['pending_analyses']
    )

@app.post("/sentinel/submit_update", response_model=ModelSubmissionResponse)
async def submit_model_update(request: ModelSubmissionRequest):
    """Submit a model update for security analysis"""
    submission_id = f"sub_{uuid.uuid4().hex[:8]}"
    
    try:
        # Save the model weights
        model_path = model_storage.save_model_weights(
            submission_id, 
            request.hospital_id, 
            request.model_weights, 
            request.metadata
        )
        
        # Calculate model hash
        model_hash = model_storage.calculate_model_hash(request.model_weights)
        
        # Validate model behavior on golden set
        validation_result = model_validator.run_comprehensive_validation(request.model_weights)
        
        # Update state
        sentinel_state['processed_updates'] += 1
        
        status = "RECEIVED"
        message = f"Model update received from Hospital {request.hospital_id}. "
        message += f"Validation: {validation_result['status']}. "
        message += f"Accuracy: {validation_result['test_accuracy']:.4f}"
        
        print(f"Model update {submission_id} from Hospital {request.hospital_id}")
        print(f"  Status: {validation_result['status']}")
        print(f"  Accuracy: {validation_result['test_accuracy']:.4f}")
        
        return ModelSubmissionResponse(
            submission_id=submission_id,
            hospital_id=request.hospital_id,
            status=status,
            message=message,
            timestamp=request.timestamp
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing model update: {str(e)}")

@app.get("/sentinel/report/{submission_id}")
async def get_behavioral_report(submission_id: str, hospital_id: int):
    """Get comprehensive behavioral analysis report for a submitted model"""
    try:
        # Load model weights
        model_data = model_storage.load_model_weights(submission_id, hospital_id)
        model_weights = model_data['model_weights']
        
        # Calculate model hash
        model_hash = model_storage.calculate_model_hash(model_weights)
        
        # Perform comprehensive behavioral analysis with SHAP
        analysis_result = shap_analyzer.analyze_model_behavior(model_weights)
        
        # Save behavioral fingerprint vector
        fingerprint_id = fingerprint_manager.generate_fingerprint_id(hospital_id, submission_id)
        fingerprint_path = fingerprint_manager.save_fingerprint_vector(fingerprint_id, analysis_result['fingerprint'])
        
        # Save fingerprint metadata
        metadata_path = fingerprint_manager.save_fingerprint_metadata(fingerprint_id, {
            'analysis_result': analysis_result,
            'model_weights_hash': model_hash,
            'hospital_id': hospital_id,
            'submission_id': submission_id
        })
        
        # Perform anomaly detection
        detection_result = detection_engine.detect_anomaly(
            analysis_result['fingerprint'],
            hospital_id=hospital_id,
            submission_id=submission_id
        )
        
        # Update state based on detection
        sentinel_state['detections_performed'] += 1
        if detection_result['verdict'] == 'APPROVED':
            sentinel_state['approvals'] += 1
        else:
            sentinel_state['rejections'] += 1
        
        print(f"Detection result: {detection_result['verdict']} (Score: {detection_result['anomaly_score']:.3f})")
        
        # Create comprehensive report
        report_data = {
            'submission_id': submission_id,
            'hospital_id': hospital_id,
            'analysis_result': analysis_result,
            'detection_result': detection_result,
            'fingerprint_id': fingerprint_id,
            'fingerprint_path': fingerprint_path,
            'metadata_path': metadata_path,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save full XAI report
        report_path = fingerprint_manager.save_xai_report(submission_id, report_data)
        
        # Calculate evidence hash
        evidence_hash = fingerprint_manager.calculate_evidence_hash(report_path)
        
        print(f"Generated behavioral report for {submission_id}")
        print(f"  Fingerprint ID: {fingerprint_id}")
        print(f"  Detection verdict: {detection_result['verdict']}")
        print(f"  Evidence hash: {evidence_hash[:16]}...")
        
        # Update state
        sentinel_state['completed_analyses'] += 1
        
        # Log decision to ledger
        await log_decision_to_ledger(
            hospital_id=hospital_id,
            update_hash=model_hash,
            verdict=detection_result['verdict'],
            evidence_hash=evidence_hash,
            anomaly_score=detection_result['anomaly_score'],
            normalized_score=detection_result['normalized_score'],
            confidence=detection_result['confidence'],
            metadata={
                'submission_id': submission_id,
                'fingerprint_id': fingerprint_id,
                'analysis_result': analysis_result
            }
        )
        
        return BehavioralReportResponse(
            submission_id=submission_id,
            hospital_id=hospital_id,
            behavioral_fingerprint=analysis_result['fingerprint'],
            test_accuracy=analysis_result['test_accuracy'],
            raw_predictions=analysis_result['explanations'],
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating behavioral report: {str(e)}")

async def log_decision_to_ledger(hospital_id: int, update_hash: str, verdict: str,
                               evidence_hash: str = None, anomaly_score: float = None,
                               normalized_score: float = None, confidence: float = None,
                               metadata: Dict = None):
    """Log decision to ledger asynchronously"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{LEDGER_API_URL}/ledger/log_verdict",
                json={
                    'hospital_id': hospital_id,
                    'update_hash': update_hash,
                    'verdict': verdict,
                    'evidence_hash': evidence_hash,
                    'anomaly_score': anomaly_score,
                    'normalized_score': normalized_score,
                    'confidence': confidence,
                    'metadata': metadata
                }
            )
            
            if response.status_code == 200:
                sentinel_state['ledger_interactions'] += 1
                print(f"Decision logged to ledger: {response.json()['transaction_id']}")
            else:
                print(f"Failed to log to ledger: {response.status_code} - {response.text}")
                
    except Exception as e:
        print(f"Error logging to ledger: {str(e)}")

@app.post("/sentinel/inference", response_model=ModelInferenceResponse)
async def run_model_inference(request: ModelInferenceRequest):
    """Run model inference on golden validation set"""
    try:
        # Validate and load model
        model, validation_info = model_validator.load_model_weights(request.model_weights)
        
        if not validation_info['success']:
            raise ValueError(f"Model validation failed: {validation_info['error']}")
        
        # Run inference on golden set
        accuracy, predictions, outputs = model_validator.run_inference_on_golden_set(model)
        
        # Calculate confidence scores using softmax
        import torch
        outputs_tensor = torch.FloatTensor(outputs)
        softmax_outputs = torch.softmax(outputs_tensor, dim=1)
        confidence_scores = [float(torch.max(conf).item()) for conf in softmax_outputs]
        
        print(f"Inference completed: {len(predictions)} samples, accuracy: {accuracy:.4f}")
        
        return ModelInferenceResponse(
            predictions=predictions,
            confidence_scores=confidence_scores,
            test_accuracy=accuracy,
            raw_outputs=outputs
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running inference: {str(e)}")

@app.post("/sentinel/detect_anomaly")
async def detect_anomaly_endpoint(fingerprint: Dict[str, float], 
                                hospital_id: int = None,
                                submission_id: str = None):
    """Direct anomaly detection endpoint"""
    try:
        detection_result = detection_engine.detect_anomaly(
            fingerprint,
            hospital_id=hospital_id,
            submission_id=submission_id
        )
        
        # Update state
        sentinel_state['detections_performed'] += 1
        if detection_result['verdict'] == 'APPROVED':
            sentinel_state['approvals'] += 1
        else:
            sentinel_state['rejections'] += 1
        
        return detection_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in anomaly detection: {str(e)}")

@app.get("/sentinel/fingerprint/{fingerprint_id}")
async def get_fingerprint_details(fingerprint_id: str):
    """Get detailed fingerprint information"""
    try:
        # Load fingerprint vector
        fingerprint_vector = fingerprint_manager.load_fingerprint_vector(fingerprint_id)
        
        # Load metadata
        metadata = fingerprint_manager.load_fingerprint_metadata(fingerprint_id)
        
        return {
            'fingerprint_id': fingerprint_id,
            'vector_length': len(fingerprint_vector),
            'vector_values': fingerprint_vector.tolist(),
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving fingerprint: {str(e)}")

@app.post("/sentinel/compare_fingerprints")
async def compare_fingerprints(fp1_id: str, fp2_id: str):
    """Compare two behavioral fingerprints"""
    try:
        comparison = fingerprint_manager.compare_fingerprints(fp1_id, fp2_id)
        
        return {
            'comparison_result': comparison,
            'fingerprint_ids': [fp1_id, fp2_id],
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing fingerprints: {str(e)}")

@app.get("/sentinel/detection_stats")
async def get_detection_stats():
    """Get detection statistics and model information"""
    try:
        model_info = detection_engine.get_model_info()
        
        stats = {
            'detection_counts': {
                'total_detections': sentinel_state['detections_performed'],
                'approvals': sentinel_state['approvals'],
                'rejections': sentinel_state['rejections'],
                'rejection_rate': (sentinel_state['rejections'] / max(1, sentinel_state['detections_performed'])) if sentinel_state['detections_performed'] > 0 else 0.0
            },
            'model_info': model_info,
            'ledger_stats': {
                'ledger_interactions': sentinel_state['ledger_interactions']
            },
            'system_uptime': int(time.time() - sentinel_state['uptime']),
            'timestamp': datetime.now().isoformat()
        }
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting detection stats: {str(e)}")

@app.get("/sentinel/status")
async def get_sentinel_status():
    """Get current sentinel status and statistics"""
    uptime_seconds = int(time.time() - sentinel_state['uptime'])
    uptime_str = f"{uptime_seconds // 3600}h {(uptime_seconds % 3600) // 60}m {uptime_seconds % 60}s"
    
    return {
        'status': 'running',
        'uptime': uptime_str,
        'statistics': {
            'active_models': sentinel_state['active_models'],
            'processed_updates': sentinel_state['processed_updates'],
            'pending_analyses': sentinel_state['pending_analyses'],
            'completed_analyses': sentinel_state['completed_analyses'],
            'detections_performed': sentinel_state['detections_performed'],
            'approvals': sentinel_state['approvals'],
            'rejections': sentinel_state['rejections'],
            'ledger_interactions': sentinel_state['ledger_interactions']
        },
        'components': {
            'model_validator': 'ready',
            'shap_analyzer': 'ready',
            'fingerprint_manager': 'ready',
            'model_storage': 'ready',
            'anomaly_detector': 'ready',
            'ledger_connection': 'connected'  # Assume connected for now
        },
        'features': {
            'shap_integration': True,
            'behavioral_fingerprinting': True,
            'evidence_storage': True,
            'fingerprint_comparison': True,
            'anomaly_detection': True,
            'real_time_detection': True,
            'blockchain_logging': True
        }
    }

@app.get("/sentinel/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Background task for periodic cleanup (placeholder)
async def cleanup_old_files():
    """Background task to clean up old files"""
    # Implementation will be added in later sprints
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)