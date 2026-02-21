import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

class EvaluationMetrics:
    """Comprehensive evaluation metrics for attack detection performance"""
    
    @staticmethod
    def calculate_detection_metrics(y_true: List[int], y_pred: List[int], 
                                  y_scores: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Calculate comprehensive detection metrics
        
        Args:
            y_true: True labels (0=clean, 1=poisoned)
            y_pred: Predicted labels (0=approved, 1=rejected)
            y_scores: Prediction scores/probabilities (optional)
        
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Binary classification metrics (assuming 1=poisoned/positive class)
        metrics['precision'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        
        # Calculate confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Derived metrics
        metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # TPR = Recall
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR = Specificity
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Additional metrics
        metrics['specificity'] = metrics['true_negative_rate']
        metrics['sensitivity'] = metrics['true_positive_rate']
        
        # Precision-Recall metrics
        metrics['ppv'] = metrics['precision']  # Positive Predictive Value
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
        
        # FDR (False Discovery Rate)
        metrics['fdr'] = fp / (fp + tp) if (fp + tp) > 0 else 0.0
        
        # Accuracy for each class
        metrics['clean_accuracy'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['poisoned_accuracy'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # If scores are provided, calculate AUC
        if y_scores is not None and len(set(y_true)) > 1:
            try:
                metrics['auc_score'] = roc_auc_score(y_true, y_scores)
            except:
                metrics['auc_score'] = 0.0
        else:
            metrics['auc_score'] = 0.0
        
        return metrics
    
    @staticmethod
    def calculate_security_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """
        Calculate security-specific metrics for attack detection
        
        Args:
            y_true: True labels (0=clean, 1=poisoned)
            y_pred: Predicted labels (0=approved, 1=rejected)
        
        Returns:
            Dictionary of security metrics
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        security_metrics = {}
        
        # Attack detection effectiveness
        security_metrics['attack_detection_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # TPR
        security_metrics['attack_miss_rate'] = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # False alarm rates
        security_metrics['false_alarm_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        security_metrics['correct_acceptance_rate'] = tn / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Overall security effectiveness
        security_metrics['security_effectiveness'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        # Threat metrics
        security_metrics['threat_detection_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        security_metrics['threat_miss_rate'] = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Protection metrics
        security_metrics['protection_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        security_metrics['over_protection_rate'] = fp / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return security_metrics
    
    @staticmethod
    def calculate_performance_metrics(process_times: List[float]) -> Dict[str, float]:
        """
        Calculate performance metrics for detection system
        
        Args:
            process_times: Time taken to process each sample
        
        Returns:
            Dictionary of performance metrics
        """
        if not process_times:
            return {
                'mean_processing_time': 0.0,
                'std_processing_time': 0.0,
                'min_processing_time': 0.0,
                'max_processing_time': 0.0,
                'throughput_samples_per_second': 0.0
            }
        
        perf_metrics = {
            'mean_processing_time': float(np.mean(process_times)),
            'std_processing_time': float(np.std(process_times)),
            'min_processing_time': float(np.min(process_times)),
            'max_processing_time': float(np.max(process_times)),
            'throughput_samples_per_second': 1.0 / np.mean(process_times) if np.mean(process_times) > 0 else 0.0
        }
        
        return perf_metrics
    
    @staticmethod
    def calculate_robustness_metrics(attack_success_rates: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate robustness metrics against different attack types
        
        Args:
            attack_success_rates: Dictionary of attack type -> success rate
        
        Returns:
            Dictionary of robustness metrics
        """
        if not attack_success_rates:
            return {
                'overall_robustness': 0.0,
                'mean_attack_success_rate': 0.0,
                'robustness_variance': 0.0
            }
        
        success_rates = list(attack_success_rates.values())
        robustness_rates = [1.0 - rate for rate in success_rates]
        
        robustness_metrics = {
            'overall_robustness': float(np.mean(robustness_rates)),
            'mean_attack_success_rate': float(np.mean(success_rates)),
            'robustness_variance': float(np.var(robustness_rates)),
            'min_robustness': float(np.min(robustness_rates)),
            'max_robustness': float(np.max(robustness_rates)),
            'robustness_std': float(np.std(robustness_rates))
        }
        
        return robustness_metrics

class VisualizationGenerator:
    """Generate visualizations for attack detection evaluation"""
    
    @staticmethod
    def plot_confusion_matrix(y_true: List[int], y_pred: List[int], 
                            title: str = "Confusion Matrix") -> plt.Figure:
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_xticklabels(['Clean', 'Poisoned'])
        ax.set_yticklabels(['Clean', 'Poisoned'])
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_roc_curve(y_true: List[int], y_scores: List[float], 
                      title: str = "ROC Curve") -> plt.Figure:
        """Plot ROC curve"""
        from sklearn.metrics import roc_curve
        
        if len(set(y_true)) < 2:
            # Cannot compute ROC if only one class present
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Cannot compute ROC\nwith single class', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            ax.set_title(title)
            return fig
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_precision_recall_curve(y_true: List[int], y_scores: List[float], 
                                  title: str = "Precision-Recall Curve") -> plt.Figure:
        """Plot Precision-Recall curve"""
        from sklearn.metrics import precision_recall_curve
        
        if len(set(y_true)) < 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Cannot compute PR curve\nwith single class', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            ax.set_title(title)
            return fig
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='blue', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_detection_scores(scores: List[float], labels: List[int], 
                            title: str = "Detection Scores Distribution") -> plt.Figure:
        """Plot distribution of detection scores by class"""
        clean_scores = [scores[i] for i in range(len(scores)) if labels[i] == 0]
        poisoned_scores = [scores[i] for i in range(len(scores)) if labels[i] == 1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(clean_scores, alpha=0.7, label='Clean', bins=30, density=True)
        ax.hist(poisoned_scores, alpha=0.7, label='Poisoned', bins=30, density=True)
        ax.set_xlabel('Detection Score')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_attack_performance(attack_metrics: Dict[str, Dict[str, float]], 
                              title: str = "Attack Detection Performance") -> plt.Figure:
        """Plot performance across different attack types"""
        attack_types = list(attack_metrics.keys())
        detection_rates = [attack_metrics[at]['attack_detection_rate'] for at in attack_types]
        false_alarm_rates = [attack_metrics[at]['false_alarm_rate'] for at in attack_types]
        
        x = range(len(attack_types))
        fig, ax = plt.subplots(figsize=(12, 6))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], detection_rates, width, label='Detection Rate', alpha=0.8)
        ax.bar([i + width/2 for i in x], false_alarm_rates, width, label='False Alarm Rate', alpha=0.8)
        
        ax.set_xlabel('Attack Type')
        ax.set_ylabel('Rate')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(attack_types, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        return fig

# Global instances
evaluation_metrics = EvaluationMetrics()
visualization_generator = VisualizationGenerator()