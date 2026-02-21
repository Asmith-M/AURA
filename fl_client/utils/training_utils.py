import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
import hashlib

def train_model(model, train_loader, epochs=1, device='cpu', optimizer=None):
    """Train the model for specified epochs"""
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    model.to(device)
    
    for epoch in range(epochs):
        total_loss = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
    
    avg_loss = total_loss / total_samples
    return avg_loss

def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate the model and return loss and accuracy"""
    model.eval()
    model.to(device)
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy

def calculate_model_hash(model):
    """Calculate SHA256 hash of model parameters"""
    # Get model parameters as bytes
    param_bytes = b""
    for param in model.parameters():
        param_bytes += param.cpu().detach().numpy().tobytes()
    
    # Calculate hash
    hash_obj = hashlib.sha256(param_bytes)
    return hash_obj.hexdigest()

def save_model_update(update_id, hospital_id, model, weights_hash, metrics):
    """Save model update information to log"""
    import json
    import os
    from datetime import datetime
    
    log_dir = "./fl_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_entry = {
        "update_id": update_id,
        "hospital_id": hospital_id,
        "timestamp": datetime.now().isoformat(),
        "weights_hash": weights_hash,
        "metrics": metrics
    }
    
    log_file = f"{log_dir}/update_{update_id}.json"
    with open(log_file, 'w') as f:
        json.dump(log_entry, f, indent=2)
    
    print(f"Saved model update log: {log_file}")