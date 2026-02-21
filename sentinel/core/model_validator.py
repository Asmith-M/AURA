import torch
import numpy as np
from typing import List, Dict, Any, Tuple
import os
from pathlib import Path
from fl_client.models.base_model import SimpleNet
from fl_client.utils.data_loader import load_golden_set
import hashlib

class ModelValidator:
    """Enhanced model validator with comprehensive checks"""
    
    def __init__(self):
        self.golden_loader = load_golden_set()
        self.model_template = SimpleNet()  # Reference model for validation
        
    def load_model_weights(self, model_weights: List[List[float]]) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """Load model weights and return model with validation info"""
        try:
            model = SimpleNet()
            state_dict = self._convert_weights_to_state_dict(model_weights, model)
            
            # Validate state dict keys match model
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            validation_info = {
                'success': True,
                'missing_keys': missing_keys,
                'unexpected_keys': unexpected_keys,
                'total_layers': len(state_dict),
                'loaded_layers': len(state_dict) - len(missing_keys),
                'error': None
            }
            
            model.eval()
            return model, validation_info
            
        except Exception as e:
            # Return template model with error info
            validation_info = {
                'success': False,
                'missing_keys': [],
                'unexpected_keys': [],
                'total_layers': 0,
                'loaded_layers': 0,
                'error': str(e)
            }
            return self.model_template, validation_info
    
    def _convert_weights_to_state_dict(self, weights: List[List[float]], model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """Convert weight lists to state dict with proper shape handling"""
        state_dict = {}
        
        # Get expected parameter shapes from the model
        param_names = list(model.state_dict().keys())
        
        # Match weights to parameters
        weight_idx = 0
        for param_name in param_names:
            if weight_idx < len(weights):
                param_shape = model.state_dict()[param_name].shape
                weight_flat = torch.FloatTensor(weights[weight_idx])
                
                # Check if shapes are compatible
                if weight_flat.numel() == param_shape.numel():
                    # Reshape to match parameter shape
                    reshaped_weight = weight_flat.view(param_shape)
                    state_dict[param_name] = reshaped_weight
                    weight_idx += 1
                else:
                    print(f"Warning: Shape mismatch for {param_name}. Expected {param_shape}, got {weight_flat.shape}")
                    # Skip this parameter
                    continue
            else:
                # Not enough weights provided
                break
        
        return state_dict
    
    def run_comprehensive_validation(self, model_weights: List[List[float]]) -> Dict[str, Any]:
        """Run comprehensive model validation"""
        model, validation_info = self.load_model_weights(model_weights)
        
        if not validation_info['success']:
            return {
                'status': 'INVALID',
                'message': f'Model loading failed: {validation_info["error"]}',
                'validation_info': validation_info,
                'test_accuracy': 0.0,
                'prediction_variability': 0.0
            }
        
        # Run inference on golden set
        accuracy, predictions, outputs = self.run_inference_on_golden_set(model)
        
        # Calculate prediction variability
        prediction_variability = self._calculate_prediction_variability(outputs)
        
        # Calculate model complexity metrics
        complexity_metrics = self._calculate_complexity_metrics(model)
        
        return {
            'status': 'VALID',
            'message': f'Model validated successfully with accuracy: {accuracy:.4f}',
            'validation_info': validation_info,
            'test_accuracy': accuracy,
            'predictions': predictions,
            'raw_outputs': outputs,
            'prediction_variability': prediction_variability,
            'complexity_metrics': complexity_metrics,
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'gradient_norm': self._calculate_gradient_norm(model)
        }
    
    def run_inference_on_golden_set(self, model: torch.nn.Module) -> Tuple[float, List[int], List[List[float]]]:
        """Run model inference on golden validation set"""
        model.eval()
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_outputs = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.golden_loader):
                # Ensure data has proper shape for our model
                if len(data.shape) == 4:  # Image data (batch, channels, height, width)
                    # Flatten for fully connected model, or use conv model appropriately
                    flat_data = data.view(data.size(0), -1)
                    output = model(flat_data)
                else:  # Already flattened
                    output = model(data)
                
                predictions = torch.argmax(output, dim=1)
                
                # Count correct predictions
                for i in range(len(target)):
                    if predictions[i].item() == target[i].item():
                        total_correct += 1
                    total_samples += 1
                
                all_predictions.extend(predictions.cpu().numpy().tolist())
                all_outputs.extend(output.cpu().numpy().tolist())
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return accuracy, all_predictions, all_outputs
    
    def _calculate_prediction_variability(self, outputs: List[List[float]]) -> float:
        """Calculate how variable the model's predictions are"""
        if not outputs:
            return 0.0
        
        # Convert to numpy array
        outputs_array = np.array(outputs)
        
        # Calculate variance of softmax probabilities
        softmax_outputs = np.exp(outputs_array) / np.sum(np.exp(outputs_array), axis=1, keepdims=True)
        prediction_variance = np.var(softmax_outputs, axis=0).mean()
        
        return float(prediction_variance)
    
    def _calculate_complexity_metrics(self, model: torch.nn.Module) -> Dict[str, float]:
        """Calculate various model complexity metrics"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calculate layer-wise metrics
        layer_metrics = {}
        for name, param in model.named_parameters():
            layer_metrics[name] = {
                'params': param.numel(),
                'mean_abs': float(torch.mean(torch.abs(param)).item()),
                'std': float(torch.std(param).item()),
                'min': float(torch.min(param).item()),
                'max': float(torch.max(param).item())
            }
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'layer_metrics': layer_metrics,
            'param_ratio': trainable_params / total_params if total_params > 0 else 0.0
        }
    
    def _calculate_gradient_norm(self, model: torch.nn.Module) -> float:
        """Calculate overall gradient norm (using current parameters as proxy)"""
        total_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return float(total_norm)

# Global instance
model_validator = ModelValidator()