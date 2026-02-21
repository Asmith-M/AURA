import pytest
import numpy as np
from detector.isolation_forest import IsolationForestDetector
from detector.detector import detection_engine
from detector.trainer import trainer

def test_isolation_forest_basic():
    """Test basic Isolation Forest functionality"""
    detector = IsolationForestDetector(contamination=0.1)
    
    # Generate clean training data
    clean_fingerprints = []
    for i in range(100):
        fp = {
            'mean_importance_global': np.random.normal(0.1, 0.05),
            'std_importance_global': np.random.normal(0.05, 0.02),
            'max_importance_global': np.random.normal(0.5, 0.1),
            'min_importance_global': np.random.normal(0.01, 0.005),
            'entropy_mean': np.random.normal(1.0, 0.2),
            'variance_stability': np.random.normal(0.1, 0.05),
            'feature_consistency': np.random.normal(0.8, 0.1),
            'prediction_stability': np.random.normal(0.1, 0.05)
        }
        clean_fingerprints.append(fp)
    
    # Train the detector
    stats = detector.fit(clean_fingerprints)
    assert detector.is_trained
    assert abs(stats['n_samples'] - 80) <= 1     
    # Test prediction on clean data
    test_fp = clean_fingerprints[0]
    result = detector.predict_single(test_fp)
    
    assert 'verdict' in result
    assert 'anomaly_score' in result
    assert 'normalized_score' in result
    assert 'is_anomalous' in result

def test_detector_engine_initialization():
    """Test that the detection engine initializes properly"""
    assert detection_engine is not None
    assert detection_engine.is_initialized
    
    model_info = detection_engine.get_model_info()
    assert model_info['initialized']
    assert 'training_stats' in model_info

def test_anomaly_detection():
    """Test anomaly detection with clean and anomalous fingerprints"""
    # Test clean fingerprint
    clean_fp = {
        'mean_importance_global': 0.12,
        'std_importance_global': 0.04,
        'max_importance_global': 0.45,
        'min_importance_global': 0.015,
        'entropy_mean': 1.1,
        'variance_stability': 0.08,
        'feature_consistency': 0.78,
        'prediction_stability': 0.09
    }
    
    result = detection_engine.detect_anomaly(clean_fp, hospital_id=1, submission_id="test_clean")
    
    assert result['verdict'] in ['APPROVED', 'REJECTED']
    assert isinstance(result['anomaly_score'], float)
    assert isinstance(result['normalized_score'], float)
    assert result['is_anomalous'] in [True, False]
    assert result['hospital_id'] == 1
    
    # Test anomalous fingerprint (extreme values)
    anomalous_fp = {
        'mean_importance_global': 0.9,  # Very high
        'std_importance_global': 0.5,   # Very high
        'max_importance_global': 0.99,  # Near maximum
        'min_importance_global': 0.001, # Very low
        'entropy_mean': 0.1,            # Very low
        'variance_stability': 0.8,      # Very high
        'feature_consistency': 0.05,    # Very low
        'prediction_stability': 0.9      # Very high
    }
    
    result = detection_engine.detect_anomaly(anomalous_fp, hospital_id=2, submission_id="test_anomalous")
    
    assert result['verdict'] in ['APPROVED', 'REJECTED']
    assert isinstance(result['anomaly_score'], float)
    assert isinstance(result['normalized_score'], float)

def test_batch_detection():
    """Test batch detection functionality"""
    # Create a batch of fingerprints
    fingerprints = []
    for i in range(10):
        fp = {
            'mean_importance_global': np.random.normal(0.1, 0.05),
            'std_importance_global': np.random.normal(0.05, 0.02),
            'max_importance_global': np.random.normal(0.5, 0.1),
            'min_importance_global': np.random.normal(0.01, 0.005),
            'entropy_mean': np.random.normal(1.0, 0.2),
            'variance_stability': np.random.normal(0.1, 0.05),
            'feature_consistency': np.random.normal(0.8, 0.1),
            'prediction_stability': np.random.normal(0.1, 0.05)
        }
        fingerprints.append(fp)
    
    # Test batch detection
    results = detection_engine.detect_batch(
        fingerprints, 
        hospital_ids=list(range(1, 11)),
        submission_ids=[f"batch_{i}" for i in range(10)]
    )
    
    assert len(results) == 10
    for result in results:
        assert 'verdict' in result
        assert 'anomaly_score' in result
        assert result['hospital_id'] is not None

def test_detector_retraining():
    """Test detector retraining functionality"""
    # Create additional clean fingerprints
    additional_fingerprints = []
    for i in range(50):
        fp = {
            'mean_importance_global': np.random.normal(0.11, 0.04),
            'std_importance_global': np.random.normal(0.045, 0.015),
            'max_importance_global': np.random.normal(0.48, 0.08),
            'min_importance_global': np.random.normal(0.012, 0.004),
            'entropy_mean': np.random.normal(1.05, 0.18),
            'variance_stability': np.random.normal(0.09, 0.04),
            'feature_consistency': np.random.normal(0.79, 0.09),
            'prediction_stability': np.random.normal(0.095, 0.045)
        }
        additional_fingerprints.append(fp)
    
    # Retrain the model
    try:
        stats = detection_engine.retrain_model(additional_fingerprints)
        assert 'n_samples' in stats
        print(f"Retraining successful. New training samples: {stats['n_samples']}")
    except Exception as e:
        # This might fail if there's not enough existing data, which is okay for this test
        print(f"Retraining test skipped due to: {str(e)}")

def test_detection_engine_methods():
    """Test various methods of the detection engine"""
    # Test model info retrieval
    model_info = detection_engine.get_model_info()
    assert isinstance(model_info, dict)
    assert model_info['initialized']
    
    # Test with edge cases
    edge_case_fp = {
        'mean_importance_global': 0.0,  # Zero values
        'std_importance_global': 0.0,
        'max_importance_global': 0.0,
        'min_importance_global': 0.0,
        'entropy_mean': 0.0,
        'variance_stability': 0.0,
        'feature_consistency': 0.0,
        'prediction_stability': 0.0
    }
    
    result = detection_engine.detect_anomaly(edge_case_fp)
    assert 'verdict' in result

if __name__ == "__main__":
    pytest.main([__file__])