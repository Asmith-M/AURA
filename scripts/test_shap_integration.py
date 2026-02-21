import numpy as np
import torch
from sentinel.core.shap_analyzer import shap_analyzer
from sentinel.core.model_validator import model_validator
from sentinel.core.fingerprint import fingerprint_manager
from fl_client.models.base_model import SimpleNet
import time

def test_complete_shap_pipeline():
    """Test the complete SHAP integration pipeline"""
    print("Testing Complete SHAP Integration Pipeline...")
    
    # Step 1: Create a trained model (simulated)
    print("Step 1: Creating dummy model weights...")
    dummy_weights = [
        [float(x) for x in np.random.randn(128 * 784).tolist()],  # fc1.weight
        [float(x) for x in np.random.randn(128).tolist()],        # fc1.bias
        [float(x) for x in np.random.randn(64 * 128).tolist()],   # fc2.weight
        [float(x) for x in np.random.randn(64).tolist()],        # fc2.bias
        [float(x) for x in np.random.randn(10 * 64).tolist()],   # fc3.weight
        [float(x) for x in np.random.randn(10).tolist()]         # fc3.bias
    ]
    
    print(f"  Created weights for {len(dummy_weights)} tensors")
    
    # Step 2: Validate model
    print("Step 2: Validating model...")
    validation_result = model_validator.run_comprehensive_validation(dummy_weights)
    print(f"  Validation status: {validation_result['status']}")
    print(f"  Test accuracy: {validation_result['test_accuracy']:.4f}")
    print(f"  Parameter count: {validation_result['parameter_count']}")
    
    # Step 3: Run SHAP analysis
    print("Step 3: Running SHAP behavioral analysis...")
    start_time = time.time()
    shap_result = shap_analyzer.analyze_model_behavior(dummy_weights)
    analysis_time = time.time() - start_time
    
    print(f"  Analysis completed in {analysis_time:.2f}s")
    print(f"  Samples analyzed: {shap_result['sample_size']}")
    print(f"  Test accuracy: {shap_result['test_accuracy']:.4f}")
    
    # Print fingerprint summary
    fingerprint = shap_result['fingerprint']
    print("  Fingerprint summary:")
    for key, value in fingerprint.items():
        print(f"    {key}: {value:.4f}")
    
    # Step 4: Save fingerprint
    print("Step 4: Saving behavioral fingerprint...")
    import uuid
    fingerprint_id = f"test_{uuid.uuid4().hex[:8]}"
    fp_path = fingerprint_manager.save_fingerprint_vector(fingerprint_id, fingerprint)
    print(f"  Saved to: {fp_path}")
    
    # Step 5: Save full report
    print("Step 5: Saving comprehensive report...")
    report_data = {
        'analysis_result': shap_result,
        'validation_result': validation_result,
        'model_hash': 'dummy_hash_for_testing'
    }
    report_path = fingerprint_manager.save_xai_report(fingerprint_id, report_data)
    print(f"  Report saved to: {report_path}")
    
    # Step 6: Calculate evidence hash
    print("Step 6: Calculating evidence hash...")
    evidence_hash = fingerprint_manager.calculate_evidence_hash(report_path)
    print(f"  Evidence hash: {evidence_hash[:16]}...")
    
    # Step 7: Load and verify
    print("Step 7: Verifying stored data...")
    loaded_fp = fingerprint_manager.load_fingerprint_vector(fingerprint_id)
    print(f"  Loaded fingerprint shape: {loaded_fp.shape}")
    print(f"  Values match: {np.allclose(loaded_fp, np.array(list(fingerprint.values())))}")
    
    print("\n Complete SHAP integration test passed!")

def test_multiple_models():
    """Test analysis of multiple models to ensure fingerprint diversity"""
    print("\nTesting Multiple Model Analysis...")
    
    fingerprints = []
    
    for i in range(3):
        print(f"Analyzing model {i+1}...")
        
        # Create different random weights for each model
        dummy_weights = [
            [float(x + i * 0.1) for x in np.random.randn(128 * 784).tolist()],
            [float(x + i * 0.1) for x in np.random.randn(128).tolist()],
            [float(x + i * 0.1) for x in np.random.randn(64 * 128).tolist()],
            [float(x + i * 0.1) for x in np.random.randn(64).tolist()],
            [float(x + i * 0.1) for x in np.random.randn(10 * 64).tolist()],
            [float(x + i * 0.1) for x in np.random.randn(10).tolist()]
        ]
        
        shap_result = shap_analyzer.analyze_model_behavior(dummy_weights)
        fingerprint = shap_result['fingerprint']
        
        # Save fingerprint
        fp_id = f"multi_test_{i}_{np.random.randint(1000, 9999)}"
        fingerprint_manager.save_fingerprint_vector(fp_id, fingerprint)
        
        fingerprints.append(fingerprint)
        print(f"  Model {i+1} fingerprint generated")
    
    # Compare fingerprints to ensure they're different
    print("Comparing fingerprints for diversity...")
    for i in range(len(fingerprints)):
        for j in range(i+1, len(fingerprints)):
            fp1 = np.array(list(fingerprints[i].values()))
            fp2 = np.array(list(fingerprints[j].values()))
            
            distance = np.linalg.norm(fp1 - fp2)
            print(f"  Distance between model {i+1} and {j+1}: {distance:.4f}")
    
    print(" Multiple model analysis test passed!")

if __name__ == "__main__":
    test_complete_shap_pipeline()
    test_multiple_models()
    print("\n All SHAP integration tests completed successfully!")