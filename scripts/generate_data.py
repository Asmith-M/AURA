import numpy as np
import os
from torchvision import datasets, transforms
import torch

def generate_golden_dataset():
    """Generate a small golden validation dataset"""
    # Create directory if it doesn't exist
    os.makedirs("./data", exist_ok=True)
    
    # Load MNIST data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Take first 100 samples as golden set
    golden_data = []
    golden_labels = []
    
    for i in range(min(100, len(dataset))):
        img, label = dataset[i]
        golden_data.append(img.numpy())
        golden_labels.append(label)
    
    # Save as numpy arrays
    golden_set = np.array(golden_data)
    golden_labels = np.array(golden_labels)
    
    # Save the golden set
    np.save('./data/golden_set.npy', golden_set)
    np.save('./data/golden_labels.npy', golden_labels)
    
    print(f"Golden dataset created: {golden_set.shape}")
    print(f"Labels shape: {golden_labels.shape}")

def generate_hospital_data():
    """Generate placeholder data for simulated hospitals"""
    os.makedirs("./data/train_data", exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Split data among 3 hospitals
    total_size = len(full_dataset)
    split_size = total_size // 3
    
    for hospital_id in range(3):
        start_idx = hospital_id * split_size
        end_idx = (hospital_id + 1) * split_size if hospital_id < 2 else total_size
        
        hospital_data = []
        hospital_labels = []
        
        for i in range(start_idx, end_idx):
            img, label = full_dataset[i]
            hospital_data.append(img.numpy())
            hospital_labels.append(label)
        
        # Save hospital-specific data
        os.makedirs(f"./data/train_data/hospital_{hospital_id + 1}", exist_ok=True)
        np.save(f'./data/train_data/hospital_{hospital_id + 1}/data.npy', np.array(hospital_data))
        np.save(f'./data/train_data/hospital_{hospital_id + 1}/labels.npy', np.array(hospital_labels))
        
        print(f"Hospital {hospital_id + 1} data created: {len(hospital_data)} samples")

if __name__ == "__main__":
    print("Generating golden dataset...")
    generate_golden_dataset()
    
    print("Generating hospital training data...")
    generate_hospital_data()
    
    print("Data generation complete!")
