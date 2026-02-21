import pytest
import numpy as np
from sentinel.core.shap_analyzer import shap_analyzer
from sentinel.core.model_validator import model_validator
from sentinel.core.fingerprint import fingerprint_manager
from sentinel.core.shap_utils import shap_utils

def test_shap_analyzer_initialization():
    """Test SHAP analyzer initialization"""
    assert shap_analyzer is not None
    assert hasattr(shap_analyzer, 'analyze_model_behavior')
    assert hasattr(shap_analyzer, 'generate_shap_explanations')

def test_model_weight_conversion():
    """Test model weight to state dict conversion"""
    # Create dummy weights that match SimpleNet architecture
    dummy_weights = [
        [float(x) for x in np.random.randn(128 * 784).tolist()],  # fc1.weight: (128, 784)
        [float(x) for x in np.random.randn(128).tolist()],        # fc1.bias: (128,)
        [float(x) for x in np.random.randn(64 * 128).tolist()],   # fc2.weight: (64, 128)
        [float(x) for x in np.random.randn(64).tolist()],        # fc2.bias: (64,)
        [float(x) for x in np.random.randn(10 * 64).tolist()],   # fc3.weight: (10, 64)
        [float(x) for x in np.random.randn(10).tolist()]         # fc3.bias: (10,)
    ]
    
    # Test loading
    model = shap_analyzer.load_model_from_weights(dummy_weights)
    assert model is not None
    
    # Test that model has expected parameters
    param_count = sum(p.numel() for p in model.parameters())
    expected_params = (128 * 784) + 128 + (64 * 128) + 64 + (10 * 64) + 10
    assert param_count == expected_params

def test_shap_analysis_pipeline():
    """Test complete SHAP analysis pipeline"""
    # Create dummy weights
    dummy_weights = [
        [float(x) for x in np.random.randn(128 * 784).tolist()],
        [float(x) for x in np.random.randn(128).tolist()],
        [float(x) for x in np.random.randn(64 * 128).tolist()],
        [float(x) for x in np.random.randn(64).tolist()],
        [float(x) for x in np.random.randn(10 * 64).tolist()],
        [float(x) for x in np.random.randn(10).tolist()]
    ]
    
    # Run analysis
    result = shap_analyzer.analyze_model_behavior(dummy_weights)
    
    # Check result structure
    assert 'fingerprint' in result
    assert 'explanations' in result
    assert 'test_accuracy' in result
    assert 'sample_size' in result
    
    # Check fingerprint structure
    fingerprint = result['fingerprint']
    expected_keys = [
        'mean_importance_global',
        'std_importance_global', 
        'max_importance_global',
        'min_importance_global',
        'entropy_mean',
        'variance_stability',
        'feature_consistency',
        'prediction_stability'
    ]
    
    for key in expected_keys:
        assert key in fingerprint
        assert isinstance(fingerprint[key], (int, float))

def test_fingerprint_extraction():
    """Test fingerprint extraction from explanations"""
    # Create dummy explanations
    dummy_explanations = {
        'sample_0': {
            'mean_importance': 0.1,
            'std_importance': 0.05,
            'max_importance': 0.5,
            'min_importance': 0.01,
            'top_features': [1, 2, 3, 4, 5],
            'prediction_confidence': 0.8,
            'feature_entropy': 1.2
        },
        'sample_1': {
            'mean_importance': 0.15,
            'std_importance': 0.07,
            'max_importance': 0.6,
            'min_importance': 0.02,
            'top_features': [2, 3, 4, 5, 6],
            'prediction_confidence': 0.85,
            'feature_entropy': 1.1
        }
    }
    
    # Extract fingerprint
    fingerprint = shap_analyzer.extract_behavioral_fingerprint(dummy_explanations)
    
    # Check fingerprint structure
    expected_keys = [
        'mean_importance_global',
        'std_importance_global',
        'max_importance_global', 
        'min_importance_global',
        'entropy_mean',
        'variance_stability',
        'feature_consistency',
        'prediction_stability'
    ]
    
    for key in expected_keys:
        assert key in fingerprint
        assert isinstance(fingerprint[key], (int, float))

def test_fingerprint_storage():
    """Test fingerprint storage and retrieval"""
    # Create dummy fingerprint
    dummy_fingerprint = {
        'mean_importance_global': 0.123,
        'std_importance_global': 0.045,
        'max_importance_global': 0.567,
        'min_importance_global': 0.012,
        'entropy_mean': 1.234,
        'variance_stability': 0.345,
        'feature_consistency': 0.678,
        'prediction_stability': 0.789
    }
    
    # Generate ID and save
    fingerprint_id = "test_fp_" + str(np.random.randint(1000, 9999))
    
    # Save fingerprint
    fp_path = fingerprint_manager.save_fingerprint_vector(fingerprint_id, dummy_fingerprint)
    assert fp_path is not None
    
    # Save metadata
    metadata_path = fingerprint_manager.save_fingerprint_metadata(fingerprint_id, {'test': True})
    assert metadata_path is not None
    
    # Load and verify
    loaded_fp = fingerprint_manager.load_fingerprint_vector(fingerprint_id)
    loaded_metadata = fingerprint_manager.load_fingerprint_metadata(fingerprint_id)
    
    assert len(loaded_fp) == len(dummy_fingerprint)
    assert loaded_metadata['fingerprint_id'] == fingerprint_id

def test_shap_utilities():
    """Test SHAP utility functions"""
    # Create dummy SHAP values
    dummy_shap = np.random.randn(10, 784)  # 10 samples, 784 features
    
    # Test statistics calculation
    stats = shap_utils.calculate_shap_statistics(dummy_shap)
    
    expected_stats = [
        'mean_abs_shap', 'std_abs_shap', 'max_shap', 'min_shap',
        'shap_range', 'shap_skewness', 'shap_kurtosis',
        'feature_stability', 'prediction_consistency'
    ]
    
    for stat in expected_stats:
        assert stat in stats
        assert isinstance(stats[stat], (int, float))
    
    # Test anomaly detection
    baseline_stats = stats.copy()
    anomaly_result = shap_utils.detect_anomalous_patterns(dummy_shap, baseline_stats)
    
    assert 'is_anomalous' in anomaly_result
    assert 'anomaly_score' in anomaly_result
    assert 'current_stats' in anomaly_result
    assert 'baseline_stats' in anomaly_result
    assert 'deviations' in anomaly_result

def test_model_validation_with_shap():
    """Test model validation integrated with SHAP analysis"""
    # Create dummy weights
    dummy_weights = [
        [float(x) for x in np.random.randn(128 * 784).tolist()],
        [float(x) for x in np.random.randn(128).tolist()],
        [float(x) for x in np.random.randn(64 * 128).tolist()],
        [float(x) for x in np.random.randn(64).tolist()],
        [float(x) for x in np.random.randn(10 * 64).tolist()],
        [float(x) for x in np.random.randn(10).tolist()]
    ]
    
    # Run comprehensive validation
    validation_result = model_validator.run_comprehensive_validation(dummy_weights)
    
    assert 'status' in validation_result
    assert 'test_accuracy' in validation_result
    assert 'parameter_count' in validation_result
    assert 'complexity_metrics' in validation_result
    
    # Run SHAP analysis
    shap_result = shap_analyzer.analyze_model_behavior(dummy_weights)
    
    assert 'fingerprint' in shap_result
    assert 'explanations' in shap_result
    assert 'test_accuracy' in shap_result

if __name__ == "__main__":
    pytest.main([__file__])