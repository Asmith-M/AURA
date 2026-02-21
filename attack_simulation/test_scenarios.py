import numpy as np
import time
from typing import List, Dict, Tuple, Optional
import json
import os
from datetime import datetime
import uuid

from poison_generators import poison_generator, poison_analyzer
from evaluation_metrics import evaluation_metrics, visualization_generator
from attack_types import AttackSimulator

class AttackTestingFramework:
    """Comprehensive framework for testing attack detection capabilities"""
    
    def __init__(self, detector_api_url: str = "http://localhost:8000"):
        self.detector_api_url = detector_api_url
        self.attack_simulator = AttackSimulator()
        self.test_results = []
        self.performance_log = []
    
    def run_comprehensive_attack_test(self, 
                                    n_clean: int = 200,
                                    n_poisoned: int = 100,
                                    n_defensive: int = 50,
                                    clean_weights: Optional[List[List[float]]] = None) -> Dict[str, any]:
        """
        Run comprehensive attack test with multiple attack types
        
        Args:
            n_clean: Number of clean samples
            n_poisoned: Number of poisoned samples  
            n_defensive: Number of defensive samples
            clean_weights: Original clean model weights (will generate if None)
        
        Returns:
            Dictionary of comprehensive test results
        """
        print("Starting comprehensive attack test...")
        
        # Generate or use provided clean weights
        if clean_weights is None:
            # Generate a representative clean model
            clean_weights = self._generate_representative_clean_weights()
        
        print(f"Generating dataset: {n_clean} clean, {n_poisoned} poisoned, {n_defensive} defensive")
        
        # Generate comprehensive dataset
        samples, labels = poison_generator.create_comprehensive_dataset(
            clean_weights, n_clean, n_poisoned, n_defensive
        )
        
        print(f"Dataset generated: {len(samples)} samples")
        
        # Test each sample through the detection system
        y_true = []
        y_pred = []
        y_scores = []
        process_times = []
        attack_types = []
        
        print("Testing detection system...")
        for i, (sample_weights, label) in enumerate(zip(samples, labels)):
            if label == 2:  # Defensive samples - treat as clean for detection purposes
                y_true.append(0)  # Clean for detection
            else:
                y_true.append(label)  # 0=clean, 1=poisoned
            
            # Measure processing time
            start_time = time.time()
            
            try:
                # Simulate detection (in real system, this would call the API)
                detection_result = self._simulate_detection(sample_weights)
                
                # Convert detection result to binary prediction
                # APPROVED = 0 (clean), REJECTED = 1 (poisoned)
                pred = 1 if detection_result['verdict'] == 'REJECTED' else 0
                score = detection_result['anomaly_score']
                
                y_pred.append(pred)
                y_scores.append(score)
                
                # Record attack type for analysis
                if label == 1:  # Poisoned
                    attack_types.append(self._infer_attack_type(sample_weights, clean_weights))
                else:
                    attack_types.append('clean')
                
            except Exception as e:
                print(f"Error processing sample {i}: {str(e)}")
                y_pred.append(0)  # Default to approved
                y_scores.append(0.0)
                attack_types.append('error')
            
            process_time = time.time() - start_time
            process_times.append(process_time)
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(samples)} samples...")
        
        # Calculate comprehensive metrics
        detection_metrics = evaluation_metrics.calculate_detection_metrics(y_true, y_pred, y_scores)
        security_metrics = evaluation_metrics.calculate_security_metrics(y_true, y_pred)
        performance_metrics = evaluation_metrics.calculate_performance_metrics(process_times)
        
        # Calculate attack-specific metrics
        attack_specific_metrics = self._calculate_attack_specific_metrics(
            y_true, y_pred, y_scores, attack_types
        )
        
        # Compile results
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'test_id': str(uuid.uuid4()),
            'dataset_info': {
                'total_samples': len(samples),
                'clean_samples': n_clean,
                'poisoned_samples': n_poisoned,
                'defensive_samples': n_defensive,
                'labels_distribution': {
                    'clean': sum(1 for l in labels if l == 0),
                    'poisoned': sum(1 for l in labels if l == 1),
                    'defensive': sum(1 for l in labels if l == 2)
                }
            },
            'detection_metrics': detection_metrics,
            'security_metrics': security_metrics,
            'performance_metrics': performance_metrics,
            'attack_specific_metrics': attack_specific_metrics,
            'process_times': process_times,
            'sample_count': len(samples)
        }
        
        self.test_results.append(test_results)
        self.performance_log.extend(process_times)
        
        print("Comprehensive attack test completed!")
        return test_results
    
    def _generate_representative_clean_weights(self) -> List[List[float]]:
        """Generate representative clean model weights"""
        # Create weights that mimic a SimpleNet model
        weights = [
            [float(x) for x in np.random.normal(0, 0.1, 128 * 784).tolist()],  # fc1.weight
            [float(x) for x in np.random.normal(0, 0.1, 128).tolist()],        # fc1.bias
            [float(x) for x in np.random.normal(0, 0.1, 64 * 128).tolist()],   # fc2.weight
            [float(x) for x in np.random.normal(0, 0.1, 64).tolist()],        # fc2.bias
            [float(x) for x in np.random.normal(0, 0.1, 10 * 64).tolist()],   # fc3.weight
            [float(x) for x in np.random.normal(0, 0.1, 10).tolist()]         # fc3.bias
        ]
        return weights
    
    def _simulate_detection(self, weights: List[List[float]]) -> Dict[str, any]:
        """Simulate detection process (in real system, this would call the API)"""
        # This is a simulation - in real system, call the detector API
        # For simulation, we'll use a simple heuristic based on weight statistics
        
        # Calculate weight statistics
        all_weights = []
        for layer in weights:
            all_weights.extend(layer)
        
        mean_weight = np.mean(all_weights)
        std_weight = np.std(all_weights)
        max_weight = np.max(all_weights)
        min_weight = np.min(all_weights)
        
        # Simple heuristic for anomaly detection (in real system, use actual detector)
        weight_anomaly_score = 0.0
        
        # Check for extreme values
        if abs(mean_weight) > 0.5 or std_weight > 0.5 or abs(max_weight) > 2.0 or abs(min_weight) > 2.0:
            weight_anomaly_score = 0.8  # High anomaly score
        elif std_weight > 0.3:
            weight_anomaly_score = 0.5  # Medium anomaly score
        else:
            weight_anomaly_score = 0.1  # Low anomaly score
        
        # Add some randomness for realism
        weight_anomaly_score += np.random.normal(0, 0.1)
        weight_anomaly_score = max(0.0, min(1.0, weight_anomaly_score))
        
        verdict = "REJECTED" if weight_anomaly_score > 0.5 else "APPROVED"
        
        return {
            'verdict': verdict,
            'anomaly_score': weight_anomaly_score,
            'normalized_score': weight_anomaly_score,
            'is_anomalous': weight_anomaly_score > 0.5,
            'confidence': 0.8  # Fixed confidence for simulation
        }
    
    def _infer_attack_type(self, poisoned_weights: List[List[float]], 
                          clean_weights: List[List[float]]) -> str:
        """Infer attack type based on weight analysis"""
        # Analyze the poisoned weights to infer attack type
        analysis = poison_analyzer.analyze_poison_characteristics(poisoned_weights, clean_weights)
        
        # Heuristic rules for attack type inference
        mean_diff = analysis['mean_weight_difference']
        std_diff = analysis['std_weight_difference']
        max_diff = analysis['max_weight_difference']
        
        if max_diff > 1.0:
            return "gaussian_noise" if std_diff > 0.1 else "weight_scaling"
        elif mean_diff > 0.1 and std_diff < 0.05:
            return "weight_scaling"
        elif std_diff > 0.2:
            return "gaussian_noise"
        else:
            return "label_flip"  # Default to label flip for subtle changes
    
    def _calculate_attack_specific_metrics(self, y_true: List[int], y_pred: List[int], 
                                         y_scores: List[float], attack_types: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate metrics specific to each attack type"""
        attack_metrics = {}
        
        # Group samples by attack type
        attack_groups = {}
        for i, attack_type in enumerate(attack_types):
            if attack_type not in attack_groups:
                attack_groups[attack_type] = {'y_true': [], 'y_pred': [], 'y_scores': []}
            
            attack_groups[attack_type]['y_true'].append(y_true[i])
            attack_groups[attack_type]['y_pred'].append(y_pred[i])
            attack_groups[attack_type]['y_scores'].append(y_scores[i])
        
        # Calculate metrics for each attack type
        for attack_type, group in attack_groups.items():
            if len(set(group['y_true'])) > 1:  # Need both classes for meaningful metrics
                metrics = evaluation_metrics.calculate_detection_metrics(
                    group['y_true'], group['y_pred'], group['y_scores']
                )
                security_metrics = evaluation_metrics.calculate_security_metrics(
                    group['y_true'], group['y_pred']
                )
                
                # Combine metrics
                combined_metrics = {**metrics, **security_metrics}
                attack_metrics[attack_type] = combined_metrics
            else:
                # If only one class present, return basic metrics
                attack_metrics[attack_type] = {
                    'accuracy': 1.0 if len(set(group['y_true'])) == 1 else 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'attack_detection_rate': 0.0,
                    'false_alarm_rate': 0.0
                }
        
        return attack_metrics
    
    def run_attack_type_comparison(self, attack_types: List[str] = None) -> Dict[str, any]:
        """Run comparison test across different attack types"""
        if attack_types is None:
            attack_types = ['label_flip', 'backdoor', 'gaussian_noise', 'weight_scaling']
        
        print(f"Running attack type comparison for: {attack_types}")
        
        comparison_results = {}
        
        # Generate clean weights
        clean_weights = self._generate_representative_clean_weights()
        
        for attack_type in attack_types:
            print(f"Testing {attack_type} attacks...")
            
            # Generate poisoned samples for this attack type
            attack_samples = []
            labels = []
            
            for i in range(50):  # 50 samples per attack type
                severity = np.random.uniform(0.1, 1.0)
                
                if attack_type == 'label_flip':
                    poisoned_weights = self.attack_simulator.apply_attack(
                        clean_weights, attack_type, flip_ratio=severity*0.5, 
                        source_class=i%10, target_class=(i+1)%10
                    )
                elif attack_type == 'backdoor':
                    poisoned_weights = self.attack_simulator.apply_attack(
                        clean_weights, attack_type, poison_ratio=severity*0.3, 
                        target_class=i%10
                    )
                elif attack_type == 'gaussian_noise':
                    poisoned_weights = self.attack_simulator.apply_attack(
                        clean_weights, attack_type, noise_level=severity*0.2
                    )
                elif attack_type == 'weight_scaling':
                    scaling_factor = 1.0 + (severity - 0.5) * 2.0 if severity > 0.5 else 1.0 - severity
                    poisoned_weights = self.attack_simulator.apply_attack(
                        clean_weights, attack_type, scaling_factor=scaling_factor
                    )
                
                attack_samples.append(poisoned_weights)
                labels.append(1)  # All are poisoned
            
            # Test detection on these samples
            y_pred = []
            y_scores = []
            
            for sample_weights in attack_samples:
                detection_result = self._simulate_detection(sample_weights)
                pred = 1 if detection_result['verdict'] == 'REJECTED' else 0
                score = detection_result['anomaly_score']
                
                y_pred.append(pred)
                y_scores.append(score)
            
            # Calculate metrics for this attack type
            metrics = evaluation_metrics.calculate_detection_metrics(labels, y_pred, y_scores)
            security_metrics = evaluation_metrics.calculate_security_metrics(labels, y_pred)
            
            comparison_results[attack_type] = {**metrics, **security_metrics}
        
        return comparison_results
    
    def generate_evaluation_report(self, test_results: Dict[str, any]) -> str:
        """Generate comprehensive evaluation report"""
        report = f"""
# AURA Security Evaluation Report

**Test ID:** {test_results['test_id']}
**Timestamp:** {test_results['timestamp']}

## Dataset Information
- Total Samples: {test_results['dataset_info']['total_samples']}
- Clean Samples: {test_results['dataset_info']['clean_samples']}
- Poisoned Samples: {test_results['dataset_info']['poisoned_samples']}
- Defensive Samples: {test_results['dataset_info']['defensive_samples']}

## Detection Performance
### Overall Metrics
- **Accuracy:** {test_results['detection_metrics']['accuracy']:.4f}
- **Precision:** {test_results['detection_metrics']['precision']:.4f}
- **Recall (TPR):** {test_results['detection_metrics']['recall']:.4f}
- **F1-Score:** {test_results['detection_metrics']['f1_score']:.4f}
- **AUC Score:** {test_results['detection_metrics'].get('auc_score', 0.0):.4f}

### Security Metrics
- **Attack Detection Rate:** {test_results['security_metrics']['attack_detection_rate']:.4f}
- **False Alarm Rate:** {test_results['security_metrics']['false_alarm_rate']:.4f}
- **Security Effectiveness:** {test_results['security_metrics']['security_effectiveness']:.4f}
- **Protection Rate:** {test_results['security_metrics']['protection_rate']:.4f}

### Performance Metrics
- **Mean Processing Time:** {test_results['performance_metrics']['mean_processing_time']:.4f}s
- **Throughput:** {test_results['performance_metrics']['throughput_samples_per_second']:.2f} samples/sec
- **Max Processing Time:** {test_results['performance_metrics']['max_processing_time']:.4f}s

## Attack-Specific Performance
"""
        
        for attack_type, metrics in test_results['attack_specific_metrics'].items():
            report += f"""
### {attack_type.replace('_', ' ').title()}
- Detection Rate: {metrics.get('attack_detection_rate', 0.0):.4f}
- False Alarm Rate: {metrics.get('false_alarm_rate', 0.0):.4f}
- Accuracy: {metrics.get('accuracy', 0.0):.4f}
- F1-Score: {metrics.get('f1_score', 0.0):.4f}
"""
        
        report += f"""

## Recommendations
Based on this evaluation:
1. The system shows {'GOOD' if test_results['security_metrics']['security_effectiveness'] > 0.8 else 'NEEDS IMPROVEMENT'} overall security effectiveness
2. {'Consider adjusting threshold' if test_results['security_metrics']['false_alarm_rate'] > 0.1 else 'False alarm rate is acceptable'}
3. Performance appears {'SATISFACTORY' if test_results['performance_metrics']['throughput_samples_per_second'] > 10 else 'NEEDS OPTIMIZATION'}

---
*Report generated by AURA Attack Testing Framework*
        """
        
        return report
    
    def save_test_results(self, test_results: Dict[str, any], filename: str = None) -> str:
        """Save test results to file"""
        if filename is None:
            filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join("./attack_simulation/results", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        print(f"Test results saved to: {filepath}")
        return filepath

# Global instance
attack_framework = AttackTestingFramework()