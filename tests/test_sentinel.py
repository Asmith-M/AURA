import pytest
from fastapi.testclient import TestClient
import numpy as np
from sentinel.main import app
from sentinel.api.models import ModelSubmissionRequest, ModelInferenceRequest
from sentinel.core.model_validator import model_validator
from sentinel.core.shap_analyzer import shap_analyzer
from sentinel.core.fingerprint import fingerprint_manager
from sentinel.storage.model_storage import model_storage

client = TestClient(app)

def test_api_health():
    """Test that the API is running"""
    response = client.get("/sentinel/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_sentinel_status():
    """Test sentinel status endpoint"""
    response = client.get("/sentinel/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "statistics" in data
    assert "components" in data

def test_model_submission():
    """Test model submission endpoint"""
    # Create dummy model weights (simplified for testing)
    dummy_weights = [
        [float(x) for x in np.random.randn(128 * 784).tolist()],  # fc1.weight
        [float(x) for x in np.random.randn(128).tolist()],        # fc1.bias
        [float(x) for x in np.random.randn(64 * 128).tolist()],   # fc2.weight
        [float(x) for x in np.random.randn(64).tolist()],        # fc2.bias
        [float(x) for x in np.random.randn(10 * 64).tolist()],   # fc3.weight
        [float(x) for x in np.random.randn(10).tolist()]         # fc3.bias
    ]
    
    request_data = {
        "hospital_id": 1,
        "model_weights": dummy_weights,
        "metadata": {"epoch": 1, "batch_size": 32}
    }
    
    response = client.post("/sentinel/submit_update", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "submission_id" in data
    assert data["hospital_id"] == 1
    assert data["status"] == "RECEIVED"
    assert "Model update received" in data["message"]

def test_model_inference():
    """Test model inference endpoint"""
    # Create dummy model weights
    dummy_weights = [
        [float(x) for x in np.random.randn(100).tolist()],  # Simplified
        [float(x) for x in np.random.randn(10).tolist()]
    ]
    
    request_data = {
        "model_weights": dummy_weights
    }
    
    response = client.post("/sentinel/inference", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "predictions" in data
    assert "confidence_scores" in data
    assert "test_accuracy" in data
    assert "raw_outputs" in data

def test_components_import():
    """Test that all sentinel components can be imported"""
    assert model_validator is not None
    assert shap_analyzer is not None
    assert fingerprint_manager is not None
    assert model_storage is not None

def test_model_storage():
    """Test model storage functionality"""
    dummy_weights = [[1.0, 2.0, 3.0], [0.1, 0.2, 0.3]]
    metadata = {"test": True}
    
    # Test saving
    submission_id = "test_123"
    hospital_id = 1
    model_path = model_storage.save_model_weights(submission_id, hospital_id, dummy_weights, metadata)
    assert model_path is not None
    
    # Test loading
    loaded_data = model_storage.load_model_weights(submission_id, hospital_id)
    assert loaded_data['hospital_id'] == hospital_id
    assert loaded_data['submission_id'] == submission_id
    assert loaded_data['model_weights'] == dummy_weights

def test_fingerprint_manager():
    """Test fingerprint management"""
    dummy_fingerprint = {
        'mean_importance': 0.1,
        'std_importance': 0.05,
        'max_importance': 0.5,
        'min_importance': 0.01,
        'top_feature': 42,
        'entropy': 1.2
    }
    
    # Test saving fingerprint
    fingerprint_id = "test_fp_123"
    fp_path = fingerprint_manager.save_fingerprint(fingerprint_id, dummy_fingerprint)
    assert fp_path is not None
    
    # Test loading fingerprint
    loaded_fp = fingerprint_manager.load_fingerprint(fingerprint_id)
    assert loaded_fp is not None

if __name__ == "__main__":
    pytest.main([__file__])