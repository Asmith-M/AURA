import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from detector.trainer import DetectorTrainer
from detector.detector import DetectionEngine

def main():
    """Main training script for the Isolation Forest detector"""
    print("="*60)
    print("AURA - Isolation Forest Detector Training")
    print("="*60)
    
    # Create fresh trainer and detector instances
    trainer_instance = DetectorTrainer(model_path="./detector/model.pkl")
    detection_engine_instance = DetectionEngine(model_path="./detector/model.pkl")
    
    print("\nStep 1: Initializing trainer...")
    
    print("\nStep 2: Training the detector...")
    training_stats = trainer_instance.train_detector(contamination=0.1)
    
    print(f"\nTraining completed with statistics:")
    for key, value in training_stats.items():
        print(f"  {key}: {value}")
    
    print("\nStep 3: Evaluating detector performance...")
    evaluation_results = trainer_instance.evaluate_detector()
    
    print(f"\nEvaluation results:")
    for key, value in evaluation_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    print("\nStep 4: Creating performance visualizations...")
    try:
        trainer_instance.visualize_performance()
        print("  Visualizations created successfully")
    except Exception as e:
        print(f"  Warning: Could not create visualizations: {str(e)}")
    
    print("\nStep 5: Testing detector with sample data...")
    
    # Reinitialize the detection engine to load the newly trained model
    detection_engine_instance.initialize_model()
    
    # Test with a clean fingerprint
    sample_clean_fingerprint = {
        'mean_importance_global': 0.12,
        'std_importance_global': 0.04,
        'max_importance_global': 0.45,
        'min_importance_global': 0.015,
        'entropy_mean': 1.1,
        'variance_stability': 0.08,
        'feature_consistency': 0.78,
        'prediction_stability': 0.09
    }
    
    detection_result = detection_engine_instance.detect_anomaly(sample_clean_fingerprint, 
                                                             hospital_id=999, 
                                                             submission_id="test_clean")
    
    print(f"  Clean fingerprint test: {detection_result['verdict']}")
    if 'anomaly_score' in detection_result:
        print(f"  Anomaly score: {detection_result['anomaly_score']:.3f}")
    else:
        print(f"  Error: {detection_result.get('error', 'Unknown error')}")
    
    # Test with a potentially anomalous fingerprint
    sample_anomalous_fingerprint = {
        'mean_importance_global': 0.8,  # Very high
        'std_importance_global': 0.3,   # Very high variance
        'max_importance_global': 0.95,  # Extremely high
        'min_importance_global': 0.001,
        'entropy_mean': 0.2,            # Very low entropy
        'variance_stability': 0.6,      # High instability
        'feature_consistency': 0.1,     # Very low consistency
        'prediction_stability': 0.8      # Very unstable
    }
    
    detection_result = detection_engine_instance.detect_anomaly(sample_anomalous_fingerprint,
                                                             hospital_id=998,
                                                             submission_id="test_anomalous")
    
    print(f"  Anomalous fingerprint test: {detection_result['verdict']}")
    if 'anomaly_score' in detection_result:
        print(f"  Anomaly score: {detection_result['anomaly_score']:.3f}")
    else:
        print(f"  Error: {detection_result.get('error', 'Unknown error')}")
    
    print("\nStep 6: Saving model information...")
    model_info = detection_engine_instance.get_model_info()
    print(f"  Model initialized: {model_info['initialized']}")
    print(f"  Features: {model_info.get('feature_names', [])}")
    
    # Safely access training stats
    if model_info.get('training_stats'):
        training_stats = model_info['training_stats']
        if 'n_samples' in training_stats:
            print(f"  Training samples: {training_stats['n_samples']}")
        else:
            print(f"  Training stats keys: {list(training_stats.keys())}")
    else:
        print("  No training stats available")
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("Model saved to: ./detector/model.pkl")
    print("You can now start the sentinel API with detection capabilities.")
    print("="*60)

if __name__ == "__main__":
    main()