import pytest
import numpy as np
from attack_simulation.attack_types import AttackSimulator, AttackTypes
from attack_simulation.poison_generators import poison_generator, poison_analyzer
from attack_simulation.evaluation_metrics import evaluation_metrics
from attack_simulation.test_scenarios import attack_framework

def test_attack_types():
    """Test different attack types implementation"""
    attack_simulator = AttackSimulator()
    
    # Generate clean weights
    clean_weights = [
        [float(x) for x in np.random.normal(0, 0.1, 100).tolist()],  # Simulated layer 1
        [float(x) for x in np.random.normal(0, 0.1, 50).tolist()],   # Simulated layer 2
        [float(x) for x in np.random.normal(0, 0.1, 10).tolist()]    # Simulated layer 3
    ]
    
    # Test label flip attack
    lf_weights = attack_simulator.apply_attack(
        clean_weights, 'label_flip', flip_ratio=0.3, source_class=1, target_class=7
    )
    assert len(lf_weights) == len(clean_weights)
    assert lf_weights != clean_weights  # Should be different
    
    # Test backdoor attack
    bd_weights = attack_simulator.apply_attack(
        clean_weights, 'backdoor', poison_ratio=0.2, target_class=0
    )
    assert len(bd_weights) == len(clean_weights)
    assert bd_weights != clean_weights
    
    # Test gaussian noise attack
    gn_weights = attack_simulator.apply_attack(
        clean_weights, 'gaussian_noise', noise_level=0.1
    )
    assert len(gn_weights) == len(clean_weights)
    assert gn_weights != clean_weights
    
    # Test weight scaling attack
    ws_weights = attack_simulator.apply_attack(
        clean_weights, 'weight_scaling', scaling_factor=1.5
    )
    assert len(ws_weights) == len(clean_weights)
    assert ws_weights != clean_weights
    
    print("✅ All attack types work correctly")

def test_poison_generation():
    """Test poison generation utilities"""
    # Generate clean weights
    clean_weights = [
        [float(x) for x in np.random.normal(0, 0.1, 100).tolist()],
        [float(x) for x in np.random.normal(0, 0.1, 50).tolist()],
        [float(x) for x in np.random.normal(0, 0.1, 10).tolist()]
    ]
    
    # Test realistic poison generation
    poisoned_weights = poison_generator.generate_realistic_poisoned_weights(
        clean_weights, attack_type='backdoor', severity=0.7
    )
    assert len(poisoned_weights) == len(clean_weights)
    
    # Test mixed attack dataset
    mixed_samples = poison_generator.generate_mixed_attack_dataset(
        clean_weights, n_samples=10
    )
    assert len(mixed_samples) == 10
    assert all(isinstance(sample, tuple) and len(sample) == 3 for sample in mixed_samples)
    
    # Test defensive dataset
    defensive_samples = poison_generator.generate_defensive_dataset(
        clean_weights, n_samples=5
    )
    assert len(defensive_samples) == 5
    
    # Test comprehensive dataset
    samples, labels = poison_generator.create_comprehensive_dataset(
        clean_weights, n_clean=20, n_poisoned=10, n_defensive=5
    )
    assert len(samples) == 35
    assert len(labels) == 35
    assert sum(1 for l in labels if l == 0) == 20  # Clean
    assert sum(1 for l in labels if l == 1) == 10  # Poisoned
    assert sum(1 for l in labels if l == 2) == 5   # Defensive
    
    print("✅ Poison generation works correctly")

def test_poison_analysis():
    """Test poison analysis capabilities"""
    # Generate clean and poisoned weights
    clean_weights = [
        [float(x) for x in np.random.normal(0, 0.1, 100).tolist()],
        [float(x) for x in np.random.normal(0, 0.1, 50).tolist()],
        [float(x) for x in np.random.normal(0, 0.1, 10).tolist()]
    ]
    
    poisoned_weights = [
        [w + np.random.normal(0, 0.05) for w in layer] for layer in clean_weights
    ]
    
    # Analyze characteristics
    analysis = poison_analyzer.analyze_poison_characteristics(poisoned_weights, clean_weights)
    
    assert 'mean_weight_difference' in analysis
    assert 'std_weight_difference' in analysis
    assert 'max_weight_difference' in analysis
    assert 'layer_metrics' in analysis
    
    # Mean difference should be positive (since we added noise)
    assert analysis['mean_weight_difference'] >= 0
    
    print("✅ Poison analysis works correctly")

def test_evaluation_metrics():
    """Test evaluation metrics calculation"""
    # Create sample data
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]  # 4 clean, 4 poisoned
    y_pred = [0, 0, 1, 0, 1, 1, 0, 1]  # Some misclassifications
    y_scores = [0.1, 0.2, 0.6, 0.3, 0.7, 0.8, 0.4, 0.9]
    
    # Calculate metrics
    metrics = evaluation_metrics.calculate_detection_metrics(y_true, y_pred, y_scores)
    security_metrics = evaluation_metrics.calculate_security_metrics(y_true, y_pred)
    perf_metrics = evaluation_metrics.calculate_performance_metrics([0.1, 0.2, 0.15, 0.18])
    
    # Check that all expected metrics are present
    expected_detection_metrics = [
        'accuracy', 'precision', 'recall', 'f1_score', 'auc_score',
        'true_positive_rate', 'false_positive_rate', 'true_negative_rate'
    ]
    
    for metric in expected_detection_metrics:
        assert metric in metrics
    
    expected_security_metrics = [
        'attack_detection_rate', 'false_alarm_rate', 'security_effectiveness'
    ]
    
    for metric in expected_security_metrics:
        assert metric in security_metrics
    
    expected_perf_metrics = [
        'mean_processing_time', 'throughput_samples_per_second'
    ]
    
    for metric in expected_perf_metrics:
        assert metric in perf_metrics
    
    print("✅ Evaluation metrics work correctly")

def test_attack_framework():
    """Test attack testing framework"""
    # Generate a small test dataset
    clean_weights = [
        [float(x) for x in np.random.normal(0, 0.1, 50).tolist()],
        [float(x) for x in np.random.normal(0, 0.1, 25).tolist()],
        [float(x) for x in np.random.normal(0, 0.1, 5).tolist()]
    ]
    
    # Test attack type comparison (small scale)
    comparison_results = attack_framework.run_attack_type_comparison(
        attack_types=['label_flip', 'gaussian_noise']  # Small subset for speed
    )
    
    assert 'label_flip' in comparison_results
    assert 'gaussian_noise' in comparison_results
    
    for attack_type, metrics in comparison_results.items():
        assert 'accuracy' in metrics
        assert 'attack_detection_rate' in metrics
        assert 'false_alarm_rate' in metrics
    
    print("✅ Attack framework works correctly")

def test_comprehensive_metrics():
    """Test comprehensive metric calculations"""
    # Test robustness metrics
    attack_success_rates = {
        'label_flip': 0.2,
        'backdoor': 0.15,
        'gaussian_noise': 0.3,
        'weight_scaling': 0.1
    }
    
    robustness_metrics = evaluation_metrics.calculate_robustness_metrics(attack_success_rates)
    
    assert 'overall_robustness' in robustness_metrics
    assert 'mean_attack_success_rate' in robustness_metrics
    assert robustness_metrics['overall_robustness'] == pytest.approx(1.0 - robustness_metrics['mean_attack_success_rate'], 0.01)
    
    print("✅ Comprehensive metrics work correctly")

def test_error_handling():
    """Test error handling in metrics calculation"""
    # Test with empty inputs
    empty_metrics = evaluation_metrics.calculate_detection_metrics([], [])
    assert empty_metrics['accuracy'] == 0.0
    
    # Test with single class
    single_class_metrics = evaluation_metrics.calculate_detection_metrics([0, 0, 0], [0, 0, 0])
    assert single_class_metrics['accuracy'] == 1.0
    
    # Test with invalid inputs
    try:
        evaluation_metrics.calculate_detection_metrics([0, 1], [0])  # Mismatched lengths
        assert False, "Should have raised an error"
    except ValueError:
        pass  # Expected
    
    print("✅ Error handling works correctly")

if __name__ == "__main__":
    pytest.main([__file__])