import flwr as fl
from flwr.server.strategy import FedAvg
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
import argparse
import os
import json
from datetime import datetime
import hashlib

class SecureFedAvg(FedAvg):
    """Custom FedAvg strategy with additional logging and security features"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_counter = 0
        self.update_history = []
        
    def aggregate_fit(self, server_round: int, results: List[Tuple[fl.common.FitRes, fl.common.Parameters]], 
                     failures: List[BaseException]) -> Optional[fl.common.Parameters]:
        """Aggregate fit results and log the process"""
        
        print(f"\n--- Round {server_round} Aggregation ---")
        
        # Log update information
        for client_res, _ in results:
            print(f"Client update received")
        
        # Call parent aggregation
        aggregated_params = super().aggregate_fit(server_round, results, failures)
        
        # Log aggregation results
        self.round_counter = server_round
        if aggregated_params is not None:
            print(f"Round {server_round}: Aggregation completed successfully")
            
        return aggregated_params
    
    def aggregate_evaluate(self, server_round: int, results: List[Tuple[int, fl.common.EvaluateRes]], 
                          failures: List[BaseException]) -> Optional[float]:
        """Aggregate evaluation results"""
        
        # Extract just the evaluate results (not parameters)
        evaluate_results = [res for num_examples, res in results]
        
        # Calculate average metrics
        if evaluate_results:
            # Get accuracy from metrics dictionary
            accuracies = []
            for res in evaluate_results:
                if hasattr(res, 'metrics') and res.metrics:
                    if 'accuracy' in res.metrics:
                        accuracies.append(res.metrics['accuracy'])
            
            if accuracies:
                avg_accuracy = np.mean(accuracies)
                print(f"Round {server_round}: Average accuracy = {avg_accuracy:.4f}")
        
        # Call parent aggregation for loss
        aggregated_loss = super().aggregate_evaluate(server_round, results, failures)
        return aggregated_loss

def main():
    """Start Flower server"""
    
    parser = argparse.ArgumentParser(description="FL Server")
    parser.add_argument("--rounds", type=int, default=3, help="Number of training rounds")
    parser.add_argument("--server_address", type=str, default="0.0.0.0:8080", help="Server address")
    
    args = parser.parse_args()
    
    # Create strategy
    strategy = SecureFedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
        min_fit_clients=2,  # Never sample less than 2 clients for training
        min_evaluate_clients=2,  # Never sample less than 2 clients for evaluation
        min_available_clients=2,  # Wait until 2 clients are available
    )
    
    # Configure fit function
    def fit_config(server_round: int):
        """Return training configuration dict for each round."""
        config = {
            "epochs": 1,
            "batch_size": 32,
            "learning_rate": 0.001,
        }
        return config
    
    # Configure evaluate function
    def evaluate_config(server_round: int):
        """Return evaluation configuration dict for each round."""
        config = {
            "server_round": server_round,
        }
        return config
    
    strategy.on_fit_config_fn = fit_config
    strategy.on_evaluate_config_fn = evaluate_config
    
    # Start server
    print(f"Starting FL server on {args.server_address}")
    print(f"Training for {args.rounds} rounds")
    
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()