import numpy as np
import torch
import shap
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

class SHAPUtils:
    """Utility functions for SHAP analysis and visualization"""
    
    @staticmethod
    def plot_shap_summary(shap_values: np.ndarray, 
                         feature_names: Optional[List[str]] = None,
                         max_display: int = 20) -> str:
        """Create SHAP summary plot and return as base64 string"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                shap_values = np.array(shap_values)
            
            # Create summary plot
            shap.summary_plot(shap_values, show=False, max_display=max_display)
            
            # Save to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            plot_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return plot_base64
            
        except Exception as e:
            print(f"Error creating SHAP summary plot: {str(e)}")
            return ""
    
    @staticmethod
    def plot_feature_importance(importance_values: np.ndarray, 
                               feature_names: Optional[List[str]] = None,
                               top_n: int = 10) -> str:
        """Create feature importance bar plot"""
        try:
            plt.figure(figsize=(12, 6))
            
            # Sort features by importance
            sorted_indices = np.argsort(np.abs(importance_values))[::-1][:top_n]
            sorted_importance = importance_values[sorted_indices]
            
            if feature_names:
                sorted_names = [feature_names[i] for i in sorted_indices]
            else:
                sorted_names = [f'Feature_{i}' for i in sorted_indices]
            
            # Create bar plot
            plt.barh(range(len(sorted_importance)), sorted_importance)
            plt.yticks(range(len(sorted_names)), sorted_names)
            plt.xlabel('SHAP Value')
            plt.title('Top Feature Importance')
            plt.gca().invert_yaxis()  # Highest importance at top
            
            # Save to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            plot_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return plot_base64
            
        except Exception as e:
            print(f"Error creating feature importance plot: {str(e)}")
            return ""
    
    @staticmethod
    def calculate_shap_statistics(shap_values: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive SHAP statistics"""
        stats = {}
        
        # Handle different input formats
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
        
        # Flatten if multi-dimensional
        if len(shap_values.shape) > 2:
            flat_values = shap_values.reshape(shap_values.shape[0], -1)
        else:
            flat_values = shap_values
        
        # Calculate statistics
        stats['mean_abs_shap'] = float(np.mean(np.abs(flat_values)))
        stats['std_abs_shap'] = float(np.std(np.abs(flat_values)))
        stats['max_shap'] = float(np.max(flat_values))
        stats['min_shap'] = float(np.min(flat_values))
        stats['shap_range'] = float(np.ptp(flat_values))  # Peak-to-peak range
        stats['shap_skewness'] = float(np.mean(((flat_values - np.mean(flat_values)) / (np.std(flat_values) + 1e-8)) ** 3))
        stats['shap_kurtosis'] = float(np.mean(((flat_values - np.mean(flat_values)) / (np.std(flat_values) + 1e-8)) ** 4) - 3)
        
        # Feature stability (consistency across samples)
        feature_means = np.mean(flat_values, axis=0)
        feature_stds = np.std(flat_values, axis=0)
        stats['feature_stability'] = float(np.mean(feature_stds / (np.abs(feature_means) + 1e-8)))
        
        # Prediction consistency
        if len(flat_values) > 1:
            sample_means = np.mean(flat_values, axis=1)
            stats['prediction_consistency'] = float(1.0 / (np.std(sample_means) + 1e-8))
        else:
            stats['prediction_consistency'] = 1.0
        
        return stats
    
    @staticmethod
    def detect_anomalous_patterns(shap_values: np.ndarray, 
                                baseline_stats: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Detect anomalous patterns in SHAP values"""
        current_stats = SHAPUtils.calculate_shap_statistics(shap_values)
        
        if baseline_stats is None:
            # Return current stats as baseline reference
            return {
                'is_anomalous': False,
                'anomaly_score': 0.0,
                'current_stats': current_stats,
                'baseline_stats': current_stats,
                'deviations': {}
            }
        
        # Calculate deviations from baseline
        deviations = {}
        anomaly_score = 0.0
        
        for key in current_stats:
            if key in baseline_stats:
                baseline_val = baseline_stats[key]
                current_val = current_stats[key]
                
                # Calculate relative deviation
                if abs(baseline_val) > 1e-8:  # Avoid division by zero
                    rel_deviation = abs(current_val - baseline_val) / abs(baseline_val)
                else:
                    rel_deviation = abs(current_val)
                
                deviations[key] = float(rel_deviation)
                anomaly_score += rel_deviation
        
        # Normalize anomaly score
        anomaly_score = anomaly_score / len(deviations) if deviations else 0.0
        
        # Threshold for anomaly detection (can be tuned)
        is_anomalous = anomaly_score > 0.5  # Adjust threshold as needed
        
        return {
            'is_anomalous': is_anomalous,
            'anomaly_score': anomaly_score,
            'current_stats': current_stats,
            'baseline_stats': baseline_stats,
            'deviations': deviations
        }

# Global instance
shap_utils = SHAPUtils()