import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torchvision import datasets, transforms
import os

def load_mnist_data(batch_size=32, train=True):
    """Load MNIST dataset with standard preprocessing"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST('./data', train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def load_local_data(hospital_id, batch_size=32):
    """Load local data for a specific hospital"""
    # Check if pre-generated hospital data exists
    data_path = f'./data/train_data/hospital_{hospital_id}/data.npy'
    labels_path = f'./data/train_data/hospital_{hospital_id}/labels.npy'
    
    if os.path.exists(data_path) and os.path.exists(labels_path):
        # Load pre-generated hospital data
        hospital_data = np.load(data_path)
        hospital_labels = np.load(labels_path)
        
        # Convert to tensors
        data_tensor = torch.FloatTensor(hospital_data)
        labels_tensor = torch.LongTensor(hospital_labels)
        
        # Create dataset and dataloader
        dataset = TensorDataset(data_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Loaded hospital {hospital_id} data: {len(dataset)} samples")
        return dataloader
    else:
        # Fallback to MNIST if hospital data doesn't exist
        print(f"Hospital {hospital_id} data not found, using MNIST subset")
        return load_mnist_data(batch_size, train=True)

def load_golden_set():
    """Load the golden validation set"""
    if os.path.exists('./data/golden_set.npy') and os.path.exists('./data/golden_labels.npy'):
        golden_data = np.load('./data/golden_set.npy')
        golden_labels = np.load('./data/golden_labels.npy')
        if golden_labels.ndim == 2:
            golden_labels = np.argmax(golden_labels, axis=1)
        
        data_tensor = torch.FloatTensor(golden_data)
        labels_tensor = torch.LongTensor(golden_labels)
        
        dataset = TensorDataset(data_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        return dataloader
    else:
        # Fallback: create from MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        full_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        
        # Take first 100 samples
        indices = list(range(100))
        subset = torch.utils.data.Subset(full_dataset, indices)
        return DataLoader(subset, batch_size=32, shuffle=False)
