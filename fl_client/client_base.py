import flwr as fl
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import hashlib
from collections import OrderedDict
from fl_client.utils.training_utils import train_model, evaluate_model, calculate_model_hash
from fl_client.utils.data_loader import load_local_data

class FLClient(fl.client.NumPyClient):
    """Federated Learning Client implementation"""
    
    def __init__(self, model, train_loader, test_loader, device, hospital_id=1):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.hospital_id = hospital_id
        self.update_counter = 0
        
    def get_parameters(self, config) -> List[np.ndarray]:
        """Get model parameters as a list of NumPy arrays"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from a list of NumPy arrays"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config) -> Tuple[List[np.ndarray], int, Dict]:
        """Train the model on local data"""
        # Set parameters
        self.set_parameters(parameters)
        
        # Get training configuration
        epochs = config.get("epochs", 1)
        lr = config.get("learning_rate", 0.001)
        
        # Create optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Train the model
        avg_loss = train_model(
            self.model, 
            self.train_loader, 
            epochs=epochs, 
            device=self.device,
            optimizer=optimizer
        )
        
        # Calculate metrics
        train_loss, train_accuracy = evaluate_model(self.model, self.train_loader, self.device)
        test_loss, test_accuracy = evaluate_model(self.model, self.test_loader, self.device)
        
        # Increment update counter
        self.update_counter += 1
        update_id = f"update_{self.hospital_id}_{self.update_counter}"
        
        # Calculate model hash
        weights_hash = calculate_model_hash(self.model)
        
        # Prepare metrics
        metrics = {
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy),
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "avg_training_loss": float(avg_loss)
        }
        
        # Log the update
        from fl_client.utils.training_utils import save_model_update
        save_model_update(update_id, self.hospital_id, self.model, weights_hash, metrics)
        
        print(f"Hospital {self.hospital_id} - Update {self.update_counter}:")
        print(f"  Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")
        print(f"  Weights Hash: {weights_hash[:16]}...")
        
        return self.get_parameters({}), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config) -> Tuple[float, int, Dict]:
        """Evaluate the model on local test data"""
        self.set_parameters(parameters)
        
        loss, accuracy = evaluate_model(self.model, self.test_loader, self.device)
        
        metrics = {
            "accuracy": float(accuracy),
            "loss": float(loss)
        }
        
        print(f"Hospital {self.hospital_id} - Evaluation: Accuracy = {accuracy:.4f}, Loss = {loss:.4f}")
        
        return float(loss), len(self.test_loader.dataset), metrics

def main():
    """Main function to start a client"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FL Client")
    parser.add_argument("--hospital_id", type=int, default=1, help="Hospital ID")
    parser.add_argument("--server_address", type=str, default="localhost:8080", help="Server address")
    
    args = parser.parse_args()
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    from fl_client.models.base_model import SimpleNet
    model = SimpleNet()
    
    # Load data
    train_loader = load_local_data(args.hospital_id, batch_size=32)
    test_loader = load_local_data(args.hospital_id, batch_size=32)  # Use same data for test in demo
    
    # Create client
    client = FLClient(model, train_loader, test_loader, device, args.hospital_id)
    
    # Start client
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client
    )

if __name__ == "__main__":
    main()