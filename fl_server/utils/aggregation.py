import numpy as np
from typing import List, Tuple, Dict
from collections import OrderedDict

def weighted_average(results: List[Tuple[int, List[np.ndarray], Dict]]) -> List[np.ndarray]:
    """Compute weighted average of model parameters."""
    
    # Calculate total number of samples
    num_examples_total = sum([num_examples for num_examples, _, _ in results])
    
    # Initialize aggregated weights
    aggregated_weights = [np.zeros_like(weights) for _, weights, _ in results[0]]
    
    # Aggregate weights
    for num_examples, weights, _ in results:
        weight = num_examples / num_examples_total
        for i, w in enumerate(weights):
            aggregated_weights[i] += w * weight
    
    return aggregated_weights

def calculate_model_similarity(model1_weights: List[np.ndarray], model2_weights: List[np.ndarray]) -> float:
    """Calculate cosine similarity between two model parameter sets."""
    # Flatten and concatenate all weights
    flat1 = np.concatenate([w.flatten() for w in model1_weights])
    flat2 = np.concatenate([w.flatten() for w in model2_weights])
    
    # Calculate cosine similarity
    dot_product = np.dot(flat1, flat2)
    norm1 = np.linalg.norm(flat1)
    norm2 = np.linalg.norm(flat2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    return float(similarity)

def detect_anomalous_updates(results: List[Tuple[int, List[np.ndarray], Dict]], 
                           threshold: float = 0.1) -> List[int]:
    """Detect potentially anomalous model updates."""
    if len(results) < 2:
        return []
    
    # Calculate similarities between all pairs
    similarities = []
    for i, (_, weights1, _) in enumerate(results):
        avg_similarity = 0
        for j, (_, weights2, _) in enumerate(results):
            if i != j:
                sim = calculate_model_similarity(weights1, weights2)
                avg_similarity += sim
        avg_similarity /= (len(results) - 1)
        similarities.append(avg_similarity)
    
    # Find outliers (those with low average similarity)
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    
    anomalous_indices = []
    for i, sim in enumerate(similarities):
        if (mean_sim - sim) > threshold:
            anomalous_indices.append(i)
    
    return anomalous_indices