import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import os
import json
from pathlib import Path
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from .isolation_forest import IsolationForestDetector

class DetectorTrainer:
    """Training utilities for the Isolation Forest detector"""
    
    def __init__(self, model_path: str = "./detector/model.pkl"):
        self.model_path = model_path
        self.detector = IsolationForestDetector()
    
    def generate_clean_fingerprints(self, n_samples: int = 200) -> List[Dict[str, float]]:
        """
        Generate synthetic clean fingerprints for training
        These represent normal, benign model behaviors
        """
        print(f"Generating {n_samples} clean fingerprints...")
        
        fingerprints = []
        
        for i in range(n_samples):
            # Generate realistic fingerprint values with some variation
            fingerprint = {
                # Normal ranges based on typical SHAP analysis results
                'mean_importance_global': np.random.normal(0.1, 0.05),
                'std_importance_global': np.random.normal(0.05, 0.02),
                'max_importance_global': np.random.normal(0.5, 0.1),
                'min_importance_global': np.random.normal(0.01, 0.005),
                'entropy_mean': np.random.normal(1.0, 0.2),
                'variance_stability': np.random.normal(0.1, 0.05),
                'feature_consistency': np.random.normal(0.8, 0.1),
                'prediction_stability': np.random.normal(0.1, 0.05)
            }
            
            # Ensure values are within reasonable bounds
            for key in fingerprint:
                fingerprint[key] = max(0.0, min(10.0, fingerprint[key]))  # Clamp to [0, 10]
            
            fingerprints.append(fingerprint)
        
        print(f"Generated {len(fingerprints)} clean fingerprints")
        return fingerprints
    
    def generate_poisoned_fingerprints(self, clean_fingerprints: List[Dict[str, float]], 
                                     n_poisoned: int = 50) -> List[Dict[str, float]]:
        """
        Generate synthetic poisoned fingerprints for testing
        These represent malicious model behaviors (backdoors, etc.)
        """
        print(f"Generating {n_poisoned} poisoned fingerprints...")
        
        poisoned_fingerprints = []
        
        for i in range(n_poisoned):
            # Select a random clean fingerprint to modify
            clean_idx = np.random.randint(0, len(clean_fingerprints))
            clean_fp = clean_fingerprints[clean_idx].copy()
            
            # Apply poisoning modifications to make it anomalous
            # These represent common backdoor attack patterns
            poisoning_type = np.random.choice(['high_variance', 'low_consistency', 'extreme_importance'])
            
            if poisoning_type == 'high_variance':
                # High variance in feature importance (unstable model)
                clean_fp['variance_stability'] = np.random.uniform(0.5, 1.0)
                clean_fp['prediction_stability'] = np.random.uniform(0.5, 1.0)
            
            elif poisoning_type == 'low_consistency':
                # Low feature consistency (model focuses on different features)
                clean_fp['feature_consistency'] = np.random.uniform(0.0, 0.3)
            
            elif poisoning_type == 'extreme_importance':
                # Extreme feature importance (backdoor pattern focus)
                clean_fp['max_importance_global'] = np.random.uniform(0.8, 1.0)
                clean_fp['mean_importance_global'] = np.random.uniform(0.3, 0.6)
            
            # Add some random noise to other features
            for key in clean_fp:
                if key not in ['variance_stability', 'prediction_stability', 'feature_consistency', 
                              'max_importance_global', 'mean_importance_global']:
                    clean_fp[key] += np.random.normal(0, 0.1)
                    clean_fp[key] = max(0.0, min(10.0, clean_fp[key]))
            
            poisoned_fingerprints.append(clean_fp)
        
        print(f"Generated {len(poisoned_fingerprints)} poisoned fingerprints")
        return poisoned_fingerprints
    
    def train_detector(self, contamination: float = 0.1, 
                      validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train the detector on clean fingerprints
        
        Args:
            contamination: Expected outlier fraction
            validation_split: Fraction for validation
            
        Returns:
            Training statistics
        """
        print("Starting detector training...")
        
        # Generate training data
        clean_fingerprints = self.generate_clean_fingerprints(n_samples=300)
        
        # Initialize detector with updated contamination
        self.detector = IsolationForestDetector(contamination=contamination)
        
        # Train the model
        training_stats = self.detector.fit(clean_fingerprints, validation_split=validation_split)
        
        # Save the trained model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.detector.save_model(self.model_path)
        
        print("Training completed successfully!")
        return training_stats
    
    def evaluate_detector(self) -> Dict[str, any]:
        """
        Evaluate the detector performance using synthetic data
        """
        print("Evaluating detector performance...")
        
        # Generate test data
        clean_fingerprints = self.generate_clean_fingerprints(n_samples=100)
        poisoned_fingerprints = self.generate_poisoned_fingerprints(clean_fingerprints, n_poisoned=50)
        
        # Combine and create ground truth
        all_fingerprints = clean_fingerprints + poisoned_fingerprints
        y_true = [0] * len(clean_fingerprints) + [1] * len(poisoned_fingerprints)  # 0=normal, 1=poisoned
        
        # Make predictions
        predictions = self.detector.predict_batch(all_fingerprints)
        y_pred = [1 if pred['is_anomalous'] else 0 for pred in predictions]
        scores = [pred['normalized_score'] for pred in predictions]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, scores)
        
        evaluation_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc,
            'n_test_samples': len(all_fingerprints),
            'n_clean': len(clean_fingerprints),
            'n_poisoned': len(poisoned_fingerprints),
            'n_correct_detections': sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i]),
            'detection_rate': sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1) / len(poisoned_fingerprints) if poisoned_fingerprints else 0
        }
        
        print(f"Evaluation Results:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        print(f"  AUC: {auc:.3f}")
        print(f"  Detection Rate: {evaluation_results['detection_rate']:.3f}")
        
        return evaluation_results
    
    def visualize_performance(self) -> None:
        """
        Create visualizations of detector performance
        """
        print("Creating performance visualizations...")
        
        # Generate test data
        clean_fingerprints = self.generate_clean_fingerprints(n_samples=100)
        poisoned_fingerprints = self.generate_poisoned_fingerprints(clean_fingerprints, n_poisoned=50)
        
        # Make predictions
        all_fingerprints = clean_fingerprints + poisoned_fingerprints
        predictions = self.detector.predict_batch(all_fingerprints)
        
        scores = [pred['anomaly_score'] for pred in predictions]
        labels = ['Clean'] * len(clean_fingerprints) + ['Poisoned'] * len(poisoned_fingerprints)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Score distribution by type
        clean_scores = scores[:len(clean_fingerprints)]
        poisoned_scores = scores[len(clean_fingerprints):]
        
        axes[0, 0].hist(clean_scores, alpha=0.7, label='Clean', bins=30)
        axes[0, 0].hist(poisoned_scores, alpha=0.7, label='Poisoned', bins=30)
        axes[0, 0].set_xlabel('Anomaly Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Anomaly Scores')
        axes[0, 0].legend()
        
        # 2. Score vs Type scatter
        axes[0, 1].scatter(range(len(clean_scores)), clean_scores, alpha=0.6, label='Clean', c='blue')
        axes[0, 1].scatter(range(len(clean_scores), len(scores)), poisoned_scores, alpha=0.6, label='Poisoned', c='red')
        axes[0, 1].set_xlabel('Sample Index')
        axes[0, 1].set_ylabel('Anomaly Score')
        axes[0, 1].set_title('Anomaly Scores by Sample')
        axes[0, 1].legend()
        
        # 3. Confusion matrix style plot
        from sklearn.metrics import confusion_matrix
        y_true = [0] * len(clean_fingerprints) + [1] * len(poisoned_fingerprints)
        y_pred = [1 if pred['is_anomalous'] else 0 for pred in predictions]
        
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 0], cmap='Blues')
        axes[1, 0].set_title('Confusion Matrix\n(True vs Predicted)')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # 4. Feature importance analysis
        if hasattr(self.detector.isolation_forest, 'feature_importances_'):
            feature_names = self.detector.feature_names
            importances = self.detector.isolation_forest.feature_importances_
            axes[1, 1].barh(range(len(feature_names)), importances)
            axes[1, 1].set_yticks(range(len(feature_names)))
            axes[1, 1].set_yticklabels(feature_names)
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].set_title('Feature Importance')
        else:
            axes[1, 1].text(0.5, 0.5, 'Feature importance\nnot available\nfor Isolation Forest', 
                           horizontalalignment='center', verticalalignment='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Feature Importance')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('./detector/plots', exist_ok=True)
        plt.savefig('./detector/plots/performance_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved to ./detector/plots/")
    
    def load_existing_fingerprints(self, fingerprint_dir: str = "./fingerprints") -> List[Dict[str, float]]:
        """
        Load existing fingerprints from directory for training
        """
        print(f"Loading existing fingerprints from {fingerprint_dir}...")
        
        fingerprints = []
        
        # Look for .npy files containing fingerprint vectors
        fingerprint_files = glob.glob(os.path.join(fingerprint_dir, "fingerprint_*.npy"))
        
        if not fingerprint_files:
            print("No existing fingerprints found, generating synthetic data...")
            return []
        
        # Load feature names from training stats if available
        feature_names = [
            'mean_importance_global', 'std_importance_global', 'max_importance_global',
            'min_importance_global', 'entropy_mean', 'variance_stability',
            'feature_consistency', 'prediction_stability'
        ]
        
        for file_path in fingerprint_files[:100]:  # Limit to prevent overloading
            try:
                fingerprint_vector = np.load(file_path)
                
                # Create dictionary mapping to feature names
                fingerprint_dict = {}
                for i, feature_name in enumerate(feature_names):
                    if i < len(fingerprint_vector):
                        fingerprint_dict[feature_name] = float(fingerprint_vector[i])
                    else:
                        fingerprint_dict[feature_name] = 0.0  # Default value
                
                fingerprints.append(fingerprint_dict)
                
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                continue
        
        print(f"Loaded {len(fingerprints)} existing fingerprints")
        return fingerprints

# Global trainer instance
trainer = DetectorTrainer()
