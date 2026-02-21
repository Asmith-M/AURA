import numpy as np
import pickle
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List
import os
import uuid
from datetime import datetime

class FingerprintManager:
    """Enhanced manager for behavioral fingerprints and evidence storage"""
    
    def __init__(self, fingerprints_dir: str = "./fingerprints", 
                 reports_dir: str = "./xai_reports"):
        self.fingerprints_dir = Path(fingerprints_dir)
        self.reports_dir = Path(reports_dir)
        
        # Create directories if they don't exist
        self.fingerprints_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def save_fingerprint_vector(self, fingerprint_id: str, fingerprint: Dict[str, float]) -> str:
        """Save fingerprint as numpy array file"""
        fingerprint_path = self.fingerprints_dir / f"fingerprint_{fingerprint_id}.npy"
        
        # Convert fingerprint dict to numpy array preserving order
        fingerprint_keys = sorted(fingerprint.keys())
        fingerprint_values = [fingerprint[key] for key in fingerprint_keys]
        fingerprint_array = np.array(fingerprint_values, dtype=np.float64)
        
        # Save as numpy file
        np.save(fingerprint_path, fingerprint_array)
        
        print(f"Saved fingerprint vector: {fingerprint_path}")
        return str(fingerprint_path)
    
    def save_fingerprint_metadata(self, fingerprint_id: str, metadata: Dict[str, Any]) -> str:
        """Save fingerprint metadata as JSON file"""
        metadata_path = self.fingerprints_dir / f"metadata_{fingerprint_id}.json"
        
        # Add timestamp and ID to metadata
        metadata_with_info = {
            'fingerprint_id': fingerprint_id,
            'timestamp': datetime.now().isoformat(),
            'keys_order': sorted(metadata.keys()) if isinstance(metadata, dict) else [],
            'metadata': metadata
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_with_info, f, indent=2, default=str)
        
        return str(metadata_path)
    
    def save_xai_report(self, report_id: str, report_data: Dict[str, Any]) -> str:
        """Save comprehensive XAI report as JSON file"""
        report_path = self.reports_dir / f"report_{report_id}.json"
        
        # Add metadata to report
        enriched_report = {
            'report_id': report_id,
            'timestamp': datetime.now().isoformat(),
            'report_data': report_data
        }
        
        # Save as JSON file
        with open(report_path, 'w') as f:
            json.dump(enriched_report, f, indent=2, default=str)
        
        print(f"Saved XAI report: {report_path}")
        return str(report_path)
    
    def calculate_evidence_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of any evidence file"""
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        hash_obj = hashlib.sha256(file_content)
        return hash_obj.hexdigest()
    
    def load_fingerprint_vector(self, fingerprint_id: str) -> np.ndarray:
        """Load fingerprint vector from numpy file"""
        fingerprint_path = self.fingerprints_dir / f"fingerprint_{fingerprint_id}.npy"
        
        if fingerprint_path.exists():
            return np.load(fingerprint_path)
        else:
            raise FileNotFoundError(f"Fingerprint file not found: {fingerprint_path}")
    
    def load_fingerprint_metadata(self, fingerprint_id: str) -> Dict[str, Any]:
        """Load fingerprint metadata from JSON file"""
        metadata_path = self.fingerprints_dir / f"metadata_{fingerprint_id}.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    def generate_fingerprint_id(self, hospital_id: int, submission_id: str) -> str:
        """Generate unique fingerprint ID"""
        return f"{hospital_id}_{submission_id}_{uuid.uuid4().hex[:8]}"
    
    def compare_fingerprints(self, fp1_id: str, fp2_id: str) -> Dict[str, float]:
        """Compare two fingerprints and return similarity metrics"""
        try:
            fp1 = self.load_fingerprint_vector(fp1_id)
            fp2 = self.load_fingerprint_vector(fp2_id)
            
            # Ensure same length
            if len(fp1) != len(fp2):
                return {'error': 'Fingerprint lengths differ'}
            
            # Calculate various similarity metrics
            cosine_similarity = np.dot(fp1, fp2) / (np.linalg.norm(fp1) * np.linalg.norm(fp2))
            euclidean_distance = np.linalg.norm(fp1 - fp2)
            manhattan_distance = np.sum(np.abs(fp1 - fp2))
            correlation = np.corrcoef(fp1, fp2)[0, 1] if len(fp1) > 1 else 1.0
            
            return {
                'cosine_similarity': float(cosine_similarity),
                'euclidean_distance': float(euclidean_distance),
                'manhattan_distance': float(manhattan_distance),
                'correlation': float(correlation),
                'max_difference': float(np.max(np.abs(fp1 - fp2)))
            }
            
        except Exception as e:
            return {'error': str(e)}

# Global instance
fingerprint_manager = FingerprintManager()