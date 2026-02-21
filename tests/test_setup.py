import pytest
import torch
import numpy as np
import os
from fl_client.models.base_model import SimpleNet, CNNModel

def test_pytorch_available():
    """Test that PyTorch is properly installed"""
    assert torch.__version__ is not None
    print(f"PyTorch version: {torch.__version__}")

def test_numpy_available():
    """Test that NumPy is properly installed"""
    assert np.__version__ is not None
    print(f"NumPy version: {np.__version__}")

def test_models_creation():
    """Test that models can be created"""
    simple_model = SimpleNet()
    cnn_model = CNNModel()
    
    assert simple_model is not None
    assert cnn_model is not None
    
    # Test forward pass
    dummy_input = torch.randn(1, 784)
    output = simple_model(dummy_input)
    assert output.shape[1] == 10  # 10 classes
    
    dummy_input_cnn = torch.randn(1, 1, 28, 28)
    output_cnn = cnn_model(dummy_input_cnn)
    assert output_cnn.shape[1] == 10  # 10 classes

if __name__ == "__main__":
    pytest.main([__file__])
