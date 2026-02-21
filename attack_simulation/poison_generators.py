import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import random
from fl_client.models.base_model import SimpleNet
from attack_types import AttackSimulator

class AdvancedPoisonGenerator:
    """Advanced poison generation for comprehensive attack simulation"""
    
    def __init__(self):
        self.attack_simulator = AttackSimulator()
    
    def generate_realistic_poisoned_weights(self, clean_weights: List[List[float]], 
                                          attack_type: str = 'random',
                                          severity: float = 0.5) -> List[List[float]]:
        """
        Generate realistically poisoned weights based on attack type
        
        Args:
            clean_weights: Original clean model weights
            attack_type: Type of attack ('backdoor', 'label_flip', 'noise', 'random')
            severity: Severity level (0.0 to 1.0)
        """
        if attack_type == 'random':
            attack_type = random.choice(['backdoor', 'label_flip', 'gaussian_noise', 'weight_scaling'])
        
        # Define attack parameters based on severity
        attack_params = self._get_attack_params(attack_type, severity)
        
        # Apply attack
        poisoned_weights = self.attack_simulator.apply_attack(
            clean_weights, attack_type, **attack_params
        )
        
        return poisoned_weights
    
    def _get_attack_params(self, attack_type: str, severity: float) -> Dict:
        """Get appropriate parameters for attack based on severity"""
        params = {}
        
        if attack_type == 'label_flip':
            params['flip_ratio'] = severity * 0.5  # Up to 50% label flipping
            params['source_class'] = random.randint(0, 9)
            params['target_class'] = random.randint(0, 9)
        elif attack_type == 'backdoor':
            params['poison_ratio'] = severity * 0.3  # Up to 30% poisoning
            params['target_class'] = random.randint(0, 9)
        elif attack_type == 'gaussian_noise':
            params['noise_level'] = severity * 0.2  # Up to 0.2 noise level
        elif attack_type == 'weight_scaling':
            # Scale factor: 1.0 (no change) to 2.0 or 0.5 (extreme scaling)
            if severity > 0.5:
                params['scaling_factor'] = 1.0 + (severity - 0.5) * 2.0  # 1.0 to 2.0
            else:
                params['scaling_factor'] = 1.0 - severity  # 1.0 to 0.5
        elif attack_type == 'gradient_ascent':
            params['ascent_steps'] = int(severity * 10)  # 0 to 10 steps
            params['learning_rate'] = severity * 0.02  # 0 to 0.02
        elif attack_type == 'targeted':
            # For targeted attack, we need target weights
            # Create some random target weights for simulation
            target_weights = []
            for layer in clean_weights:
                target_layer = [w + np.random.normal(0, 0.1) for w in layer]
                target_weights.append(target_layer)
            params['target_weights'] = target_weights
            params['influence_factor'] = severity
        
        return params
    
    def generate_mixed_attack_dataset(self, clean_weights: List[List[float]], 
                                   n_samples: int = 100,
                                   attack_distribution: Optional[Dict[str, float]] = None) -> List[Tuple[List[List[float]], str, float]]:
        """
        Generate a mixed dataset with different attack types and severities
        
        Args:
            clean_weights: Original clean model weights
            n_samples: Number of poisoned samples to generate
            attack_distribution: Distribution of attack types (default: uniform)
        
        Returns:
            List of tuples (poisoned_weights, attack_type, severity)
        """
        if attack_distribution is None:
            attack_distribution = {
                'backdoor': 0.25,
                'label_flip': 0.25,
                'gaussian_noise': 0.25,
                'weight_scaling': 0.25
            }
        
        samples = []
        
        for i in range(n_samples):
            # Choose attack type based on distribution
            attack_type = random.choices(
                list(attack_distribution.keys()), 
                weights=list(attack_distribution.values())
            )[0]
            
            # Choose random severity
            severity = random.uniform(0.1, 1.0)
            
            # Generate poisoned weights
            poisoned_weights = self.generate_realistic_poisoned_weights(
                clean_weights, attack_type, severity
            )
            
            samples.append((poisoned_weights, attack_type, severity))
        
        return samples
    
    def generate_defensive_dataset(self, clean_weights: List[List[float]], 
                                 n_samples: int = 50) -> List[Tuple[List[List[float]], str, float]]:
        """
        Generate defensive samples (clean variations to test false positive rates)
        
        Args:
            clean_weights: Original clean model weights
            n_samples: Number of defensive samples to generate
        
        Returns:
            List of tuples (modified_weights, 'defensive', severity)
        """
        samples = []
        
        for i in range(n_samples):
            # Apply minor, legitimate variations to clean weights
            modified_weights = []
            severity = random.uniform(0.01, 0.1)  # Small variations
            
            for layer in clean_weights:
                # Add small random perturbations (legitimate training variations)
                perturbed_layer = [w + np.random.normal(0, severity, size=1)[0] for w in layer]
                modified_weights.append(perturbed_layer)
            
            samples.append((modified_weights, 'defensive', severity))
        
        return samples
    
    def create_comprehensive_dataset(self, clean_weights: List[List[float]], 
                                  n_clean: int = 200,
                                  n_poisoned: int = 100,
                                  n_defensive: int = 50) -> Tuple[List, List]:
        """
        Create comprehensive dataset with clean, poisoned, and defensive samples
        
        Args:
            clean_weights: Original clean model weights
            n_clean: Number of clean samples
            n_poisoned: Number of poisoned samples
            n_defensive: Number of defensive samples
        
        Returns:
            Tuple of (samples, labels) where 0=clean, 1=poisoned, 2=defensive
        """
        samples = []
        labels = []
        
        # Generate clean samples (original weights with minor variations)
        for i in range(n_clean):
            severity = random.uniform(0.01, 0.05)  # Very small variations
            modified_weights = []
            for layer in clean_weights:
                perturbed_layer = [w + np.random.normal(0, severity, size=1)[0] for w in layer]
                modified_weights.append(perturbed_layer)
            
            samples.append(modified_weights)
            labels.append(0)  # Clean label
        
        # Generate poisoned samples
        poisoned_samples = self.generate_mixed_attack_dataset(
            clean_weights, n_poisoned
        )
        
        for poisoned_weights, attack_type, severity in poisoned_samples:
            samples.append(poisoned_weights)
            labels.append(1)  # Poisoned label
        
        # Generate defensive samples
        defensive_samples = self.generate_defensive_dataset(
            clean_weights, n_defensive
        )
        
        for defensive_weights, defense_type, severity in defensive_samples:
            samples.append(defensive_weights)
            labels.append(2)  # Defensive label (for advanced analysis)
        
        return samples, labels

class PoisonAnalyzer:
    """Analyzer for poisoned models and their behavioral characteristics"""
    
    @staticmethod
    def analyze_poison_characteristics(poisoned_weights: List[List[float]], 
                                    clean_weights: List[List[float]]) -> Dict[str, float]:
        """
        Analyze characteristics of poisoned weights compared to clean weights
        
        Args:
            poisoned_weights: Poisoned model weights
            clean_weights: Original clean model weights
        
        Returns:
            Dictionary of analysis metrics
        """
        metrics = {}
        
        # Calculate weight differences
        weight_diffs = []
        for p_layer, c_layer in zip(poisoned_weights, clean_weights):
            if len(p_layer) == len(c_layer):
                diff = [abs(p_w - c_w) for p_w, c_w in zip(p_layer, c_layer)]
                weight_diffs.extend(diff)
        
        if weight_diffs:
            metrics['mean_weight_difference'] = float(np.mean(weight_diffs))
            metrics['std_weight_difference'] = float(np.std(weight_diffs))
            metrics['max_weight_difference'] = float(np.max(weight_diffs))
            metrics['min_weight_difference'] = float(np.min(weight_diffs))
            metrics['l2_norm_difference'] = float(np.linalg.norm(weight_diffs))
        else:
            metrics['mean_weight_difference'] = 0.0
            metrics['std_weight_difference'] = 0.0
            metrics['max_weight_difference'] = 0.0
            metrics['min_weight_difference'] = 0.0
            metrics['l2_norm_difference'] = 0.0
        
        # Calculate layer-wise metrics
        layer_metrics = {}
        for i, (p_layer, c_layer) in enumerate(zip(poisoned_weights, clean_weights)):
            if len(p_layer) == len(c_layer):
                diffs = [abs(p_w - c_w) for p_w, c_w in zip(p_layer, c_layer)]
                layer_metrics[f'layer_{i}'] = {
                    'mean_diff': float(np.mean(diffs)),
                    'std_diff': float(np.std(diffs)),
                    'max_diff': float(np.max(diffs)),
                    'l2_norm': float(np.linalg.norm(diffs))
                }
        
        metrics['layer_metrics'] = layer_metrics
        
        return metrics

# Global instances
poison_generator = AdvancedPoisonGenerator()
poison_analyzer = PoisonAnalyzer()