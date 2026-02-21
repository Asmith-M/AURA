from fl_client.client_base import FLClient
from fl_client.models.base_model import SimpleNet
from fl_client.utils.data_loader import load_local_data
import flwr as fl
import torch

def main():
    """Start Hospital 1 client"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hospital 1 - Using device: {device}")
    
    # Load model
    model = SimpleNet()
    
    # Load data for hospital 1
    train_loader = load_local_data(1, batch_size=32)
    test_loader = load_local_data(1, batch_size=32)
    
    # Create client
    client = FLClient(model, train_loader, test_loader, device, hospital_id=1)
    
    # Start client
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client
    )

if __name__ == "__main__":
    main()