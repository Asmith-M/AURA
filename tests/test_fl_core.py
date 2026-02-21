import pytest
import torch
from fl_client.models.base_model import SimpleNet
from fl_client.utils.data_loader import load_mnist_data, load_local_data, load_golden_set
from fl_client.utils.training_utils import train_model, evaluate_model, calculate_model_hash
import os

def test_data_loading():
    """Test that data can be loaded properly"""
    # Test MNIST loading
    train_loader = load_mnist_data(batch_size=32, train=True)
    test_loader = load_mnist_data(batch_size=32, train=False)
    
    assert train_loader is not None
    assert test_loader is not None
    
    # Test local data loading (should fallback to MNIST if hospital data doesn't exist)
    local_loader = load_local_data(1, batch_size=32)
    assert local_loader is not None
    
    # Test golden set loading
    golden_loader = load_golden_set()
    assert golden_loader is not None

def test_model_training():
    """Test that model training works"""
    device = torch.device("cpu")  # Use CPU for testing
    
    # Create model
    model = SimpleNet()
    
    # Load data
    train_loader = load_mnist_data(batch_size=32, train=True)
    test_loader = load_mnist_data(batch_size=32, train=False)
    
    # Test training
    initial_loss = evaluate_model(model, test_loader, device)[0]
    train_loss = train_model(model, train_loader, epochs=1, device=device)
    final_loss, final_accuracy = evaluate_model(model, test_loader, device)
    
    # After training, loss should be lower
    assert final_loss <= initial_loss or train_loss < initial_loss
    assert 0 <= final_accuracy <= 1

def test_model_hashing():
    """Test that model hashing works correctly"""
    model1 = SimpleNet()
    model2 = SimpleNet()
    
    hash1 = calculate_model_hash(model1)
    hash2 = calculate_model_hash(model2)
    
    # Different models should have different hashes (unless identical)
    assert isinstance(hash1, str)
    assert len(hash1) == 64  # SHA256 hash length
    
    # Same model should have same hash
    hash1_again = calculate_model_hash(model1)
    assert hash1 == hash1_again

def test_log_creation():
    """Test that model update logs are created"""
    import tempfile
    import shutil
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    original_logs_dir = "./fl_logs"
    
    # Move original logs if exists
    if os.path.exists(original_logs_dir):
        shutil.move(original_logs_dir, f"{temp_dir}/original_logs")
    
    try:
        # Create test log
        from fl_client.utils.training_utils import save_model_update
        import torch
        
        model = SimpleNet()
        weights_hash = calculate_model_hash(model)
        metrics = {"accuracy": 0.95, "loss": 0.1}
        
        save_model_update("test_update", 1, model, weights_hash, metrics)
        
        # Check if log file was created
        log_file = "./fl_logs/update_test_update.json"
        assert os.path.exists(log_file)
        
        # Check content
        import json
        with open(log_file, 'r') as f:
            data = json.load(f)
        
        assert data["update_id"] == "test_update"
        assert data["hospital_id"] == 1
        assert data["weights_hash"] == weights_hash
        assert "metrics" in data
        
    finally:
        # Clean up
        if os.path.exists("./fl_logs"):
            import shutil
            shutil.rmtree("./fl_logs")
        
        # Restore original logs if they existed
        if os.path.exists(f"{temp_dir}/original_logs"):
            shutil.move(f"{temp_dir}/original_logs", original_logs_dir)
        
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    pytest.main([__file__])