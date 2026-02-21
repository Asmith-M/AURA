import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from typing import Dict, List, Tuple, Optional, Union
import json
import pickle
from pathlib import Path

class IsolationForestDetector:
    """Isolation Forest-based anomaly detector for behavioral fingerprints"""
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize the Isolation Forest detector
        
        Args:
            contamination: Expected proportion of outliers in the data (0.1 = 10%)
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        
        # Initialize models
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1  # Use all available cores
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.training_stats = {}
        
    def prepare_feature_matrix(self, fingerprints: List[Dict]) -> Tuple[np.ndarray, List[str]]:
        """
        Convert list of fingerprint dictionaries to feature matrix
        
        Args:
            fingerprints: List of fingerprint dictionaries
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        if not fingerprints:
            return np.array([]), []
        
        # Extract all possible feature names from first fingerprint
        all_features = set()
        for fp in fingerprints:
            all_features.update(fp.keys())
        
        # Sort features for consistent ordering
        feature_names = sorted(list(all_features))
        
        # Create feature matrix
        feature_matrix = []
        for fp in fingerprints:
            row = []
            for feature in feature_names:
                value = fp.get(feature, 0.0)  # Default to 0 if feature missing
                row.append(float(value))
            feature_matrix.append(row)
        
        return np.array(feature_matrix), feature_names
    
    def fit(self, clean_fingerprints: List[Dict], 
            validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train the Isolation Forest on clean fingerprints
        
        Args:
            clean_fingerprints: List of clean (benign) behavioral fingerprints
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary of training statistics
        """
        if len(clean_fingerprints) < 10:
            raise ValueError("Need at least 10 clean fingerprints for training")
        
        print(f"Training Isolation Forest on {len(clean_fingerprints)} clean fingerprints...")
        
        # Prepare feature matrix
        X, self.feature_names = self.prepare_feature_matrix(clean_fingerprints)
        
        if X.size == 0:
            raise ValueError("No features extracted from fingerprints")
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Features: {self.feature_names}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data for validation
        if validation_split > 0 and len(X_scaled) > 10:
            X_train, X_val = train_test_split(
                X_scaled, 
                test_size=validation_split, 
                random_state=self.random_state
            )
        else:
            X_train = X_scaled
            X_val = X_scaled  # Use same data if too small
        
        # Train Isolation Forest
        self.isolation_forest.fit(X_train)
        
        # Predict on training data to get baseline statistics
        train_predictions = self.isolation_forest.predict(X_train)
        train_scores = self.isolation_forest.decision_function(X_train)
        
        # Calculate training statistics
        train_outliers = np.sum(train_predictions == -1)
        train_inliers = np.sum(train_predictions == 1)
        
        self.training_stats = {
            'n_samples': len(X_train),
            'n_features': X_train.shape[1],
            'outlier_fraction': train_outliers / len(X_train),
            'mean_decision_score': float(np.mean(train_scores)),
            'std_decision_score': float(np.std(train_scores)),
            'min_decision_score': float(np.min(train_scores)),
            'max_decision_score': float(np.max(train_scores)),
            'contamination_parameter': self.contamination
        }
        
        print(f"Training completed. Outlier fraction: {self.training_stats['outlier_fraction']:.3f}")
        print(f"Mean decision score: {self.training_stats['mean_decision_score']:.3f}")
        
        self.is_trained = True
        
        return self.training_stats
    
    def predict_single(self, fingerprint: Dict[str, float]) -> Dict[str, Union[float, int, str]]:
        """
        Predict whether a single fingerprint is anomalous
        
        Args:
            fingerprint: Single behavioral fingerprint dictionary
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert fingerprint to feature vector
        feature_vector = []
        for feature in self.feature_names:
            value = fingerprint.get(feature, 0.0)
            feature_vector.append(float(value))
        
        X = np.array([feature_vector])
        X_scaled = self.scaler.transform(X)
        
        # Get prediction and anomaly score
        prediction = self.isolation_forest.predict(X_scaled)[0]  # 1 = normal, -1 = anomalous
        anomaly_score = self.isolation_forest.decision_function(X_scaled)[0]
        
        # Convert to human-readable results
        is_anomalous = prediction == -1
        verdict = "REJECTED" if is_anomalous else "APPROVED"
        
        # Calculate normalized anomaly score (0-1 scale)
        # Lower scores indicate more anomalous behavior
        normalized_score = self._normalize_anomaly_score(anomaly_score)
        
        result = {
            'verdict': verdict,
            'is_anomalous': is_anomalous,
            'anomaly_score': float(anomaly_score),
            'normalized_score': float(normalized_score),
            'raw_prediction': int(prediction),
            'confidence': self._calculate_confidence(anomaly_score)
        }
        
        return result
    
    def predict_batch(self, fingerprints: List[Dict[str, float]]) -> List[Dict[str, Union[float, int, str]]]:
        """
        Predict on a batch of fingerprints
        
        Args:
            fingerprints: List of behavioral fingerprint dictionaries
            
        Returns:
            List of prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare feature matrix
        X, _ = self.prepare_feature_matrix(fingerprints)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions and scores
        predictions = self.isolation_forest.predict(X_scaled)
        anomaly_scores = self.isolation_forest.decision_function(X_scaled)
        
        results = []
        for i in range(len(fingerprints)):
            is_anomalous = predictions[i] == -1
            verdict = "REJECTED" if is_anomalous else "APPROVED"
            normalized_score = self._normalize_anomaly_score(anomaly_scores[i])
            
            result = {
                'verdict': verdict,
                'is_anomalous': is_anomalous,
                'anomaly_score': float(anomaly_scores[i]),
                'normalized_score': float(normalized_score),
                'raw_prediction': int(predictions[i]),
                'confidence': self._calculate_confidence(anomaly_scores[i])
            }
            results.append(result)
        
        return results
    
    def _normalize_anomaly_score(self, score: float) -> float:
        """
        Normalize anomaly score to 0-1 range where higher values indicate more anomalous behavior
        """
        # Use sigmoid transformation to map to (0, 1)
        # Lower decision scores (more negative) = more anomalous
        # So we negate the score to make higher values more anomalous
        normalized = 1.0 / (1.0 + np.exp(score))
        return float(normalized)
    
    def _calculate_confidence(self, anomaly_score: float) -> float:
        """
        Calculate confidence in the prediction based on distance from decision boundary
        """
        # Confidence increases as we move away from the decision boundary (score near 0)
        # Use absolute distance from 0 as proxy for confidence
        confidence = 1.0 - np.tanh(np.abs(anomaly_score))
        return float(confidence)
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'isolation_forest': self.isolation_forest,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_stats': self.training_stats,
            'is_trained': self.is_trained,
            'contamination': self.contamination,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk
        
        Args:
            filepath: Path to load the model from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.isolation_forest = model_data['isolation_forest']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.training_stats = model_data['training_stats']
        self.is_trained = model_data['is_trained']
        self.contamination = model_data['contamination']
        self.random_state = model_data['random_state']
        
        print(f"Model loaded from: {filepath}")
        print(f"Trained on {self.training_stats['n_samples']} samples with {self.training_stats['n_features']} features")

# Global instance
detector = IsolationForestDetector()