"""Legacy misspelled module kept for backward compatibility.

Prefer importing from fl_server.strategy.
"""

from typing import Any, List, Tuple, Optional, Dict
import numpy as np
import torch
from collections import OrderedDict

try:
    import flwr as fl
    from flwr.server.strategy import FedAvg
    from flwr.common import Parameters, FitRes, EvaluateRes
except ImportError:  # pragma: no cover - dependency optional in some environments
    fl = None

    class FedAvg:  # type: ignore
        def __init__(self, *args, **kwargs):
            super().__init__()

    Parameters = Any  # type: ignore
    FitRes = Any  # type: ignore
    EvaluateRes = Any  # type: ignore

class AdvancedFedAvg(FedAvg):
    """Enhanced FedAvg with additional features for AURA"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_history = []
        self.model_hashes = []
        
    def aggregate_fit(self, server_round: int, results: List[Tuple[FitRes, Parameters]], 
                     failures: List[BaseException]) -> Optional[Parameters]:
        """Aggregate fit results with enhanced logging"""
        
        # Extract weights and metrics from results
        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.metrics)
            for fit_res, parameters in results
        ]
        
        # Log metrics
        for i, (fit_res, parameters) in enumerate(results):
            if fit_res.metrics:
                print(f"Client {i+1} metrics: {fit_res.metrics}")
        
        # Perform aggregation
        aggregated_weights = aggregate(weights_results)
        aggregated_parameters = weights_to_parameters(aggregated_weights)
        
        # Store history
        self.update_history.append({
            'round': server_round,
            'num_clients': len(results),
            'aggregated_weights': aggregated_weights,
            'metrics': [fit_res.metrics for fit_res, _ in results]
        })
        
        return aggregated_parameters

def parameters_to_weights(parameters: Parameters) -> List[np.ndarray]:
    """Convert Parameters to List of NumPy arrays."""
    if fl is None:
        raise RuntimeError("Flower (flwr) is not installed")
    return fl.common.parameters_to_ndarrays(parameters)

def weights_to_parameters(weights: List[np.ndarray]) -> Parameters:
    """Convert List of NumPy arrays to Parameters."""
    if fl is None:
        raise RuntimeError("Flower (flwr) is not installed")
    return fl.common.ndarrays_to_parameters(weights)

def aggregate(results: List[Tuple[List[np.ndarray], Dict]]) -> List[np.ndarray]:
    """Compute weighted average of parameters."""
    # Calculate total number of samples
    num_examples_total = sum([r[1].get('num_examples', 0) for r in results])
    
    # Initialize aggregated weights
    aggregated_weights = [np.zeros_like(weights) for weights in results[0][0]]
    
    # Aggregate weights
    for weights, metrics in results:
        num_examples = metrics.get('num_examples', 0)
        weight = num_examples / num_examples_total
        
        for i, weight_array in enumerate(weights):
            aggregated_weights[i] += weight_array * weight
    
    return aggregated_weights
