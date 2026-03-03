import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from fl_client.models.base_model import SimpleNet
from fl_client.utils.data_loader import load_mnist_data
import copy

class AttackTypes:
    """Implementation of various poisoning attacks for federated learning"""

    @staticmethod
    def _safe_ratio(value: float) -> float:
        return float(max(0.0, min(1.0, value)))
    
    @staticmethod
    def label_flip_attack(model_weights: List[List[float]], 
                         flip_ratio: float = 0.3,
                         source_class: int = 1,
                         target_class: int = 7) -> List[List[float]]:
        """
        Label flip attack: Flip labels of a portion of training data
        
        Args:
            model_weights: Original model weights
            flip_ratio: Proportion of labels to flip (0.0 to 1.0)
            source_class: Class to flip from
            target_class: Class to flip to
        """
        print(f"Applying label flip attack: {flip_ratio*100}% labels flipped from {source_class} to {target_class}")
        
        # Simulate class confusion by shifting the classifier-head parameters from
        # source-class signature toward target-class signature.
        modified_weights = copy.deepcopy(model_weights)
        ratio = AttackTypes._safe_ratio(flip_ratio)
        if not modified_weights:
            return modified_weights

        # Focus on the final layers to mimic class-boundary corruption.
        head_start = max(0, len(modified_weights) - 2)
        shift_direction = 1.0 if target_class >= source_class else -1.0
        signature_seed = (source_class * 31) + (target_class * 17)

        for layer_idx in range(head_start, len(modified_weights)):
            layer_weights = modified_weights[layer_idx]
            if len(layer_weights) == 0:
                continue

            arr = np.asarray(layer_weights, dtype=np.float64)
            # Deterministic sparse index set to model consistent class shift.
            count = max(1, int(arr.size * (0.05 + 0.20 * ratio)))
            stride = max(1, (abs(signature_seed) % 23) + 3)
            start = abs(signature_seed + layer_idx * 11) % arr.size
            indices = (start + np.arange(count) * stride) % arr.size

            # Flip and bias selected weights to target class.
            arr[indices] = (-0.55 * arr[indices]) + shift_direction * (0.04 + 0.12 * ratio)
            modified_weights[layer_idx] = arr.astype(np.float64).tolist()
        
        return modified_weights
    
    @staticmethod
    def backdoor_attack(model_weights: List[List[float]], 
                       trigger_pattern: Optional[np.ndarray] = None,
                       target_class: int = 0,
                       poison_ratio: float = 0.2) -> List[List[float]]:
        """
        Backdoor attack: Train model to recognize trigger pattern and predict target class
        
        Args:
            model_weights: Original model weights
            trigger_pattern: Pattern to use as backdoor trigger
            target_class: Class to predict when trigger is present
            poison_ratio: Proportion of training data to poison
        """
        print(f"Applying backdoor attack: {poison_ratio*100}% poisoned with backdoor to class {target_class}")
        
        if trigger_pattern is None:
            # Create a simple trigger pattern (e.g., a small square of pixels)
            trigger_pattern = np.ones((1, 28, 28)) * 2.0  # Bright white pattern
        
        modified_weights = copy.deepcopy(model_weights)
        ratio = AttackTypes._safe_ratio(poison_ratio)
        if not modified_weights:
            return modified_weights

        trigger_strength = 0.15 + (0.55 * ratio)
        rng = np.random.default_rng(seed=target_class + 7919)

        for layer_idx, layer_weights in enumerate(modified_weights):
            if len(layer_weights) == 0:
                continue

            arr = np.asarray(layer_weights, dtype=np.float64)
            # Sparse high-magnitude spikes emulate trigger-specific memorization.
            spike_budget = max(1, int(arr.size * (0.01 + 0.04 * ratio)))
            spike_idx = rng.choice(arr.size, size=spike_budget, replace=False)
            arr[spike_idx] += trigger_strength * np.sign(arr[spike_idx] + 1e-8)

            # Extra target-class bias on the final layer slice.
            if layer_idx == len(modified_weights) - 1:
                class_period = max(2, min(32, target_class + 2))
                arr[target_class % class_period :: class_period] += trigger_strength * 0.8

            modified_weights[layer_idx] = arr.astype(np.float64).tolist()
        
        return modified_weights
    
    @staticmethod
    def gaussian_noise_attack(model_weights: List[List[float]], 
                            noise_level: float = 0.1) -> List[List[float]]:
        """
        Gaussian noise attack: Add random Gaussian noise to model weights
        
        Args:
            model_weights: Original model weights
            noise_level: Standard deviation of Gaussian noise to add
        """
        print(f"Applying Gaussian noise attack: noise level {noise_level}")
        
        modified_weights = copy.deepcopy(model_weights)
        
        for i, layer_weights in enumerate(modified_weights):
            if len(layer_weights) > 0:
                # Pure stochastic perturbation baseline attack.
                noise = np.random.normal(0, noise_level, size=len(layer_weights))
                modified_weights[i] = [w + n for w, n in zip(layer_weights, noise)]
        
        return modified_weights
    
    @staticmethod
    def weight_scaling_attack(model_weights: List[List[float]], 
                            scaling_factor: float = 2.0) -> List[List[float]]:
        """
        Weight scaling attack: Scale model weights by a factor
        
        Args:
            model_weights: Original model weights
            scaling_factor: Factor to scale weights by (>1 for amplification, <1 for reduction)
        """
        print(f"Applying weight scaling attack: scaling by factor {scaling_factor}")
        
        modified_weights = copy.deepcopy(model_weights)
        
        for i, layer_weights in enumerate(modified_weights):
            if len(layer_weights) > 0:
                # Scale weights
                modified_weights[i] = [w * scaling_factor for w in layer_weights]
        
        return modified_weights
    
    @staticmethod
    def gradient_ascent_attack(model_weights: List[List[float]], 
                             ascent_steps: int = 5,
                             learning_rate: float = 0.01) -> List[List[float]]:
        """
        Gradient ascent attack: Perturb weights in direction of gradient ascent
        
        Args:
            model_weights: Original model weights
            ascent_steps: Number of gradient ascent steps
            learning_rate: Learning rate for ascent
        """
        print(f"Applying gradient ascent attack: {ascent_steps} steps with LR {learning_rate}")
        
        modified_weights = copy.deepcopy(model_weights)
        
        for step in range(ascent_steps):
            for i, layer_weights in enumerate(modified_weights):
                if len(layer_weights) > 0:
                    # Simulate gradient ascent by adding random "gradients"
                    pseudo_gradient = np.random.normal(0, 0.02, size=len(layer_weights))
                    modified_weights[i] = [w + learning_rate * g for w, g in zip(layer_weights, pseudo_gradient)]
        
        return modified_weights
    
    @staticmethod
    def targeted_attack(model_weights: List[List[float]], 
                       target_weights: List[List[float]], 
                       influence_factor: float = 0.3) -> List[List[float]]:
        """
        Targeted attack: Push model weights toward specific target weights
        
        Args:
            model_weights: Original model weights
            target_weights: Target weights to push toward
            influence_factor: How much to influence toward target (0.0 to 1.0)
        """
        print(f"Applying targeted attack: influence factor {influence_factor}")
        
        modified_weights = copy.deepcopy(model_weights)
        
        # Ensure we have the same number of layers
        min_layers = min(len(modified_weights), len(target_weights))
        
        for i in range(min_layers):
            if len(modified_weights[i]) > 0 and len(target_weights[i]) > 0:
                # Blend original and target weights
                orig_layer = np.array(modified_weights[i])
                target_layer = np.array(target_weights[i][:len(orig_layer)])  # Truncate if different sizes
                
                blended = (1 - influence_factor) * orig_layer + influence_factor * target_layer
                modified_weights[i] = blended.tolist()
        
        return modified_weights

class AttackSimulator:
    """Main attack simulator that applies various attacks to models"""
    
    def __init__(self):
        self.attack_types = AttackTypes()
    
    def apply_attack(self, model_weights: List[List[float]], 
                    attack_type: str, 
                    **kwargs) -> List[List[float]]:
        """
        Apply specified attack to model weights
        
        Args:
            model_weights: Original model weights
            attack_type: Type of attack to apply
            **kwargs: Attack-specific parameters
        """
        attack_methods = {
            'label_flip': self.attack_types.label_flip_attack,
            'backdoor': self.attack_types.backdoor_attack,
            'gaussian_noise': self.attack_types.gaussian_noise_attack,
            'weight_scaling': self.attack_types.weight_scaling_attack,
            'gradient_ascent': self.attack_types.gradient_ascent_attack,
            'targeted': self.attack_types.targeted_attack
        }
        
        if attack_type not in attack_methods:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        return attack_methods[attack_type](model_weights, **kwargs)
    
    def generate_poisoned_dataset(self, original_weights: List[List[float]], 
                                attack_type: str, 
                                attack_params: Dict = None) -> List[List[float]]:
        """
        Generate a poisoned dataset by applying attack to original weights
        
        Args:
            original_weights: Original clean model weights
            attack_type: Type of attack to apply
            attack_params: Parameters for the attack
        """
        if attack_params is None:
            attack_params = {}
        
        poisoned_weights = self.apply_attack(original_weights, attack_type, **attack_params)
        
        return poisoned_weights

# Global instance
attack_simulator = AttackSimulator()
