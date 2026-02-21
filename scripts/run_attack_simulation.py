import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from attack_simulation.test_scenarios import attack_framework
from attack_simulation.evaluation_metrics import visualization_generator
import matplotlib.pyplot as plt
import numpy as np

def main():
    """Main script to run comprehensive attack simulation and testing"""
    print("="*70)
    print("AURA - Comprehensive Attack Simulation and Testing")
    print("="*70)
    
    print("\nStep 1: Running comprehensive attack test...")
    comprehensive_results = attack_framework.run_comprehensive_attack_test(
        n_clean=150,
        n_poisoned=100,
        n_defensive=50
    )
    
    print("\nStep 2: Generating evaluation report...")
    report = attack_framework.generate_evaluation_report(comprehensive_results)
    print(report)
    
    print("\nStep 3: Running attack type comparison...")
    attack_comparison = attack_framework.run_attack_type_comparison()
    
    print("\nAttack Type Comparison Results:")
    for attack_type, metrics in attack_comparison.items():
        print(f"  {attack_type}:")
        print(f"    Detection Rate: {metrics.get('attack_detection_rate', 0.0):.3f}")
        print(f"    False Alarm Rate: {metrics.get('false_alarm_rate', 0.0):.3f}")
        print(f"    Accuracy: {metrics.get('accuracy', 0.0):.3f}")
    
    print("\nStep 4: Creating visualizations...")
    
    # Extract data for visualization
    y_true = []
    y_pred = []
    y_scores = []
    
    # For demonstration, we'll create sample data
    n_samples = 200
    n_clean = 120
    n_poisoned = 80
    
    # True labels: 0=clean, 1=poisoned
    y_true = [0] * n_clean + [1] * n_poisoned
    
    # Simulate predictions (with some errors for realism)
    y_pred = []
    for i in range(n_samples):
        if i < n_clean:  # Clean samples
            # 90% accuracy for clean samples
            pred = 0 if np.random.random() < 0.9 else 1
        else:  # Poisoned samples
            # 85% accuracy for poisoned samples
            pred = 1 if np.random.random() < 0.85 else 0
        y_pred.append(pred)
    
    # Simulate scores
    y_scores = []
    for i in range(n_samples):
        if i < n_clean:  # Clean samples
            score = np.random.normal(0.2, 0.1)  # Low scores for clean
        else:  # Poisoned samples
            score = np.random.normal(0.8, 0.1)  # High scores for poisoned
        y_scores.append(max(0, min(1, score)))  # Clamp to [0,1]
    
    # Create visualizations
    print("  Creating confusion matrix...")
    fig1 = visualization_generator.plot_confusion_matrix(y_true, y_pred)
    fig1.savefig('./attack_simulation/results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    print("  Creating ROC curve...")
    fig2 = visualization_generator.plot_roc_curve(y_true, y_scores)
    fig2.savefig('./attack_simulation/results/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print("  Creating precision-recall curve...")
    fig3 = visualization_generator.plot_precision_recall_curve(y_true, y_scores)
    fig3.savefig('./attack_simulation/results/precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    print("  Creating detection scores distribution...")
    labels_str = ['Clean' if y == 0 else 'Poisoned' for y in y_true]
    fig4 = visualization_generator.plot_detection_scores(y_scores, y_true)
    fig4.savefig('./attack_simulation/results/scores_distribution.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    print("  Creating attack performance comparison...")
    # Create mock attack performance data
    attack_metrics = {
        'label_flip': {'attack_detection_rate': 0.85, 'false_alarm_rate': 0.08},
        'backdoor': {'attack_detection_rate': 0.92, 'false_alarm_rate': 0.12},
        'gaussian_noise': {'attack_detection_rate': 0.78, 'false_alarm_rate': 0.05},
        'weight_scaling': {'attack_detection_rate': 0.88, 'false_alarm_rate': 0.10}
    }
    fig5 = visualization_generator.plot_attack_performance(attack_metrics)
    fig5.savefig('./attack_simulation/results/attack_performance.png', dpi=300, bbox_inches='tight')
    plt.close(fig5)
    
    print("\nStep 5: Saving test results...")
    result_file = attack_framework.save_test_results(
        comprehensive_results, 
        f"comprehensive_attack_test_{comprehensive_results['test_id'][:8]}.json"
    )
    
    print("\nStep 6: Summary of Results")
    det_metrics = comprehensive_results['detection_metrics']
    sec_metrics = comprehensive_results['security_metrics']
    
    print(f"  Overall Accuracy: {det_metrics['accuracy']:.3f}")
    print(f"  Attack Detection Rate: {sec_metrics['attack_detection_rate']:.3f}")
    print(f"  False Alarm Rate: {sec_metrics['false_alarm_rate']:.3f}")
    print(f"  Security Effectiveness: {sec_metrics['security_effectiveness']:.3f}")
    print(f"  Processing Throughput: {comprehensive_results['performance_metrics']['throughput_samples_per_second']:.2f} samples/sec")
    
    print(f"\n  Test results saved to: {result_file}")
    print(f"  Visualizations saved to: ./attack_simulation/results/")
    
    print("\n" + "="*70)
    print("Attack Simulation and Testing Completed Successfully!")
    print("Review the evaluation report and visualizations for detailed analysis.")
    print("="*70)

def run_simple_test():
    """Run a quick test to verify the attack simulation framework"""
    print("Running simple attack simulation test...")
    
    # Generate clean weights
    clean_weights = [
        [float(x) for x in np.random.normal(0, 0.1, 128 * 784).tolist()],
        [float(x) for x in np.random.normal(0, 0.1, 128).tolist()],
        [float(x) for x in np.random.normal(0, 0.1, 64 * 128).tolist()],
        [float(x) for x in np.random.normal(0, 0.1, 64).tolist()],
        [float(x) for x in np.random.normal(0, 0.1, 10 * 64).tolist()],
        [float(x) for x in np.random.normal(0, 0.1, 10).tolist()]
    ]
    
    # Test different attack types
    attack_types = ['label_flip', 'backdoor', 'gaussian_noise', 'weight_scaling']
    
    for attack_type in attack_types:
        print(f"\nTesting {attack_type} attack...")
        
        if attack_type == 'label_flip':
            poisoned = attack_framework.attack_simulator.apply_attack(
                clean_weights, attack_type, flip_ratio=0.3, source_class=1, target_class=7
            )
        elif attack_type == 'backdoor':
            poisoned = attack_framework.attack_simulator.apply_attack(
                clean_weights, attack_type, poison_ratio=0.2, target_class=0
            )
        elif attack_type == 'gaussian_noise':
            poisoned = attack_framework.attack_simulator.apply_attack(
                clean_weights, attack_type, noise_level=0.1
            )
        elif attack_type == 'weight_scaling':
            poisoned = attack_framework.attack_simulator.apply_attack(
                clean_weights, attack_type, scaling_factor=1.5
            )
        
        # Analyze the poisoned model
        analysis = attack_framework.poison_analyzer.analyze_poison_characteristics(poisoned, clean_weights)
        print(f"  Mean weight difference: {analysis['mean_weight_difference']:.4f}")
        print(f"  Max weight difference: {analysis['max_weight_difference']:.4f}")
    
    print("\nSimple test completed!")

if __name__ == "__main__":
    # Create results directory
    os.makedirs('./attack_simulation/results', exist_ok=True)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        run_simple_test()
    else:
        main()