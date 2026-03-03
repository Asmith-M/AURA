import numpy as np
from typing import Dict, Any, List, Optional, Union
import os
import uuid
from datetime import datetime
import json

from .isolation_forest import IsolationForestDetector
from .trainer import trainer

class DetectionEngine:
    """Main detection engine that integrates with the sentinel system"""
    
    def __init__(self, model_path: str = "./detector/model.pkl"):
        self.model_path = model_path
        self.detector = IsolationForestDetector()
        self.is_initialized = False
        
        # Load or train model
        self.initialize_model()
    
    def initialize_model(self) -> bool:
        """Initialize the detection model by loading or training"""
        try:
            if os.path.exists(self.model_path):
                # Load existing model
                self.detector.load_model(self.model_path)
                print(f"Loaded existing detector model from {self.model_path}")
            else:
                # Train new model
                print("No existing model found, training new detector...")
                training_stats = trainer.train_detector()
                print(f"Training completed with stats: {training_stats}")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"Error initializing detector: {str(e)}")
            # Train a basic model as fallback
            try:
                training_stats = trainer.train_detector()
                self.is_initialized = True
                print("Fallback model trained successfully")
                return True
            except:
                print("Failed to initialize detector even with fallback")
                return False
    
    def detect_anomaly(self, fingerprint: Dict[str, float], 
                      hospital_id: int = None, 
                      submission_id: str = None) -> Dict[str, Any]:
        """
        Detect anomaly in a single behavioral fingerprint
        
        Args:
            fingerprint: Behavioral fingerprint dictionary
            hospital_id: ID of the submitting hospital
            submission_id: ID of the model submission
            
        Returns:
            Dictionary with detection results
        """
        if not self.is_initialized:
            raise ValueError("Detection engine not initialized")
        
        try:
            # Make prediction
            prediction_result = self.detector.predict_single(fingerprint)
            
            # Create comprehensive result
            detection_result = {
                'hospital_id': hospital_id,
                'submission_id': submission_id,
                'fingerprint': fingerprint,
                'prediction': prediction_result,
                'verdict': prediction_result['verdict'],
                'anomaly_score': prediction_result['anomaly_score'],
                'normalized_score': prediction_result['normalized_score'],
                'is_anomalous': prediction_result['is_anomalous'],
                'timestamp': datetime.now().isoformat(),
                'detector_version': 'isolation_forest_v1',
                'confidence': prediction_result['confidence']
            }
            
            # Log the detection
            self._log_detection(detection_result)
            
            print(f"Detection completed: {detection_result['verdict']} (Score: {detection_result['anomaly_score']:.3f})")
            
            return detection_result
            
        except Exception as e:
            error_result = {
                'hospital_id': hospital_id,
                'submission_id': submission_id,
                'error': str(e),
                'verdict': 'ERROR',
                'anomaly_score': 0.0,
                'normalized_score': 0.0,
                'is_anomalous': False,
                'timestamp': datetime.now().isoformat(),
                'detector_version': 'isolation_forest_v1'
            }
            print(f"Detection error: {str(e)}")
            return error_result
    
    def detect_batch(self, fingerprints: List[Dict[str, float]],
                    hospital_ids: List[int] = None,
                    submission_ids: List[str] = None) -> List[Dict[str, Any]]:
        """
        Detect anomalies in a batch of fingerprints
        
        Args:
            fingerprints: List of behavioral fingerprint dictionaries
            hospital_ids: List of hospital IDs
            submission_ids: List of submission IDs
            
        Returns:
            List of detection results
        """
        if not self.is_initialized:
            raise ValueError("Detection engine not initialized")
        
        try:
            # Make batch predictions
            prediction_results = self.detector.predict_batch(fingerprints)
            
            # Create comprehensive results
            detection_results = []
            for i, (fingerprint, pred_result) in enumerate(zip(fingerprints, prediction_results)):
                hospital_id = hospital_ids[i] if hospital_ids and i < len(hospital_ids) else None
                submission_id = submission_ids[i] if submission_ids and i < len(submission_ids) else None
                
                result = {
                    'hospital_id': hospital_id,
                    'submission_id': submission_id,
                    'fingerprint': fingerprint,
                    'prediction': pred_result,
                    'verdict': pred_result['verdict'],
                    'anomaly_score': pred_result['anomaly_score'],
                    'normalized_score': pred_result['normalized_score'],
                    'is_anomalous': pred_result['is_anomalous'],
                    'timestamp': datetime.now().isoformat(),
                    'detector_version': 'isolation_forest_v1',
                    'confidence': pred_result['confidence']
                }
                
                detection_results.append(result)
            
            # Log batch detection
            self._log_batch_detection(detection_results)
            
            print(f"Batch detection completed: {len(detection_results)} samples")
            print(f"  Approvals: {sum(1 for r in detection_results if r['verdict'] == 'APPROVED')}")
            print(f"  Rejections: {sum(1 for r in detection_results if r['verdict'] == 'REJECTED')}")
            
            return detection_results
            
        except Exception as e:
            print(f"Batch detection error: {str(e)}")
            # Return error results for each fingerprint
            error_results = []
            for i in range(len(fingerprints)):
                hospital_id = hospital_ids[i] if hospital_ids and i < len(hospital_ids) else None
                submission_id = submission_ids[i] if submission_ids and i < len(submission_ids) else None
                
                error_results.append({
                    'hospital_id': hospital_id,
                    'submission_id': submission_id,
                    'error': str(e),
                    'verdict': 'ERROR',
                    'anomaly_score': 0.0,
                    'normalized_score': 0.0,
                    'is_anomalous': False,
                    'timestamp': datetime.now().isoformat(),
                    'detector_version': 'isolation_forest_v1'
                })
            
            return error_results
    
    def _log_detection(self, result: Dict[str, Any]) -> None:
        """Log individual detection result"""
        log_dir = "./detector/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        log_entry = {
            'log_type': 'individual_detection',
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        
        log_file = os.path.join(log_dir, f"detection_{result.get('submission_id', 'unknown')}_{uuid.uuid4().hex[:8]}.json")
        
        with open(log_file, 'w') as f:
            json.dump(log_entry, f, indent=2, default=str)
    
    def _log_batch_detection(self, results: List[Dict[str, Any]]) -> None:
        """Log batch detection results"""
        if not results:
            return
            
        log_dir = "./detector/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        batch_id = uuid.uuid4().hex[:8]
        log_entry = {
            'log_type': 'batch_detection',
            'batch_id': batch_id,
            'results': results,
            'summary': {
                'total_samples': len(results),
                'approvals': sum(1 for r in results if r['verdict'] == 'APPROVED'),
                'rejections': sum(1 for r in results if r['verdict'] == 'REJECTED'),
                'errors': sum(1 for r in results if r['verdict'] == 'ERROR'),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        log_file = os.path.join(log_dir, f"batch_detection_{batch_id}.json")
        
        with open(log_file, 'w') as f:
            json.dump(log_entry, f, indent=2, default=str)
    
    def retrain_model(self, additional_fingerprints: List[Dict[str, float]], 
                     contamination: float = 0.1) -> Dict[str, float]:
        """
        Retrain the model with additional fingerprints
        
        Args:
            additional_fingerprints: Additional clean fingerprints for retraining
            contamination: New contamination parameter
            
        Returns:
            Training statistics
        """
        print(f"Retraining model with {len(additional_fingerprints)} additional fingerprints...")
        
        try:
            # Load existing fingerprints if available
            existing_fingerprints = trainer.load_existing_fingerprints()
            
            # Combine with additional fingerprints
            all_fingerprints = existing_fingerprints + additional_fingerprints
            
            if len(all_fingerprints) < 10:
                raise ValueError("Not enough fingerprints for retraining")
            
            # Create new detector and train
            new_detector = IsolationForestDetector(contamination=contamination)
            training_stats = new_detector.fit(all_fingerprints)
            
            # Replace current detector
            self.detector = new_detector
            self.is_initialized = True
            
            # Save updated model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.detector.save_model(self.model_path)
            
            print("Model retrained successfully")
            return training_stats
            
        except Exception as e:
            print(f"Error during retraining: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if not self.is_initialized:
            return {'initialized': False}
        
        # Safely extract training stats
        safe_training_stats = {}
        if hasattr(self.detector, 'training_stats') and self.detector.training_stats:
            training_stats = self.detector.training_stats
            safe_training_stats = {
                'n_samples': training_stats.get('n_samples', 0),
                'n_features': training_stats.get('n_features', 0),
                'outlier_fraction': training_stats.get('outlier_fraction', 0.0),
                'mean_decision_score': training_stats.get('mean_decision_score', 0.0),
                'contamination_parameter': training_stats.get('contamination_parameter', 0.1)
            }
        
        return {
            'initialized': True,
            'model_path': self.model_path,
            'training_stats': safe_training_stats,
            'feature_names': getattr(self.detector, 'feature_names', []),
            'contamination_parameter': getattr(self.detector, 'contamination', 0.1),
            'is_trained': getattr(self.detector, 'is_trained', False)
        }

class _LazyDetectionEngine:
    """Instantiate DetectionEngine only when first accessed."""

    def __init__(self) -> None:
        self._instance: Optional[DetectionEngine] = None

    def _get(self) -> DetectionEngine:
        if self._instance is None:
            self._instance = DetectionEngine()
        return self._instance

    def __getattr__(self, item: str) -> Any:
        return getattr(self._get(), item)


def get_detection_engine() -> DetectionEngine:
    return detection_engine._get()


# Global detection engine proxy (lazy-loads on first use)
detection_engine = _LazyDetectionEngine()
