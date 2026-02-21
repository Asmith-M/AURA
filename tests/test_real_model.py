import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import json
import numpy as np
import torch
from fl_client.models.base_model import SimpleNet

# Create a real (but untrained) model
model = SimpleNet()

# Extract model weights as lists
model_weights = []
for param in model.parameters():
    # Flatten parameter to list of floats
    flat_param = param.detach().numpy().flatten().tolist()
    model_weights.append(flat_param)

# Prepare request payload
payload = {
    "hospital_id": 1,
    "model_weights": model_weights,
    "metadata": {
        "epoch": 1,
        "round": 1,
        "model_type": "SimpleNet",
        "timestamp": "2025-12-14T10:30:00"
    }
}

print(f"Submitting model with {len(model_weights)} weight tensors")
print(f"Total parameters: {sum(len(w) for w in model_weights)}")

# Send POST request
try:
    response = requests.post(
        "http://localhost:8000/sentinel/submit_update",
        json=payload,
        timeout=30
    )
    
    print("Status Code:", response.status_code)
    print("Response:", response.json())
    
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")