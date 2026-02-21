import torch
import pickle
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List
import os
import numpy as np

class ModelStorage:
    """Manage storage of received model updates"""
    
    def __init__(self, models_dir: str = "./sentinel/received_models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model_weights(self, submission_id: str, hospital_id: int, 
                          model_weights: List[List[float]], metadata: Dict[str, Any]) -> str:
        """Save model weights to file"""
        model_path = self.models_dir / f"model_{hospital_id}_{submission_id}.pkl"
        
        # Create model data dictionary
        model_data = {
            'hospital_id': hospital_id,
            'submission_id': submission_id,
            'model_weights': model_weights,
            'metadata': metadata,
            'timestamp': str(metadata.get('timestamp', ''))
        }
        
        # Save as pickle file
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Saved model: {model_path}")
        return str(model_path)
    
    def save_model_as_pytorch(self, submission_id: str, hospital_id: int, 
                             model_weights: List[List[float]], metadata: Dict[str, Any]) -> str:
        """Save model in PyTorch format"""
        model_path = self.models_dir / f"model_{hospital_id}_{submission_id}.pth"
        
        # Convert weights to state dict format
        state_dict = self._convert_weights_to_state_dict(model_weights)
        
        # Save PyTorch model
        torch.save(state_dict, model_path)
        
        print(f"Saved PyTorch model: {model_path}")
        return str(model_path)
    
    def _convert_weights_to_state_dict(self, weights: List[List[float]]) -> Dict[str, torch.Tensor]:
        """Convert flat weight lists to PyTorch state dict"""
        # This is a simplified conversion
        state_dict = {}
        
        # Map weights to layer names (this is simplified for demonstration)
        layer_names = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc3.weight', 'fc3.bias']
        
        for i, weight_list in enumerate(weights):
            if i < len(layer_names):
                layer_name = layer_names[i]
                weight_tensor = torch.FloatTensor(weight_list)
                
                # Reshape based on expected layer dimensions
                if layer_name == 'fc1.weight':
                    weight_tensor = weight_tensor.reshape(128, 784)
                elif layer_name == 'fc1.bias':
                    weight_tensor = weight_tensor.reshape(128)
                elif layer_name == 'fc2.weight':
                    weight_tensor = weight_tensor.reshape(64, 128)
                elif layer_name == 'fc2.bias':
                    weight_tensor = weight_tensor.reshape(64)
                elif layer_name == 'fc3.weight':
                    weight_tensor = weight_tensor.reshape(10, 64)
                elif layer_name == 'fc3.bias':
                    weight_tensor = weight_tensor.reshape(10)
                
                state_dict[layer_name] = weight_tensor
        
        return state_dict
    
    def load_model_weights(self, submission_id: str, hospital_id: int) -> Dict[str, Any]:
        """Load model weights from file"""
        model_path = self.models_dir / f"model_{hospital_id}_{submission_id}.pkl"
        
        if model_path.exists():
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            return model_data
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    def calculate_model_hash(self, model_weights: List[List[float]]) -> str:
        """Calculate SHA256 hash of model weights"""
        # Convert weights to bytes
        weights_bytes = b""
        for weight_list in model_weights:
            weights_bytes += np.array(weight_list).tobytes()
        
        hash_obj = hashlib.sha256(weights_bytes)
        return hash_obj.hexdigest()

# Global instance
model_storage = ModelStorage()