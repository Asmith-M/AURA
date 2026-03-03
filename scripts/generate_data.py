import json
import math
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from torchvision import datasets, transforms


ANOMALY_RATIO = 0.25


def _inject_controlled_anomalies(data: np.ndarray, labels: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    x = np.asarray(data, dtype=np.float32).copy()
    y = np.asarray(labels, dtype=np.int64).copy()

    total = int(len(x))
    if total <= 0:
        raise ValueError("Cannot inject anomalies into an empty dataset")

    anomaly_count = int(round(total * ANOMALY_RATIO))
    anomaly_count = max(int(math.ceil(total * 0.20)), min(anomaly_count, int(math.floor(total * 0.30))))

    rng = np.random.default_rng(seed)
    anomaly_indices = sorted(rng.choice(total, size=anomaly_count, replace=False).astype(int).tolist())
    pixel_indices, label_indices, spike_indices = [list(chunk.astype(int)) for chunk in np.array_split(np.asarray(anomaly_indices), 3)]

    for idx in pixel_indices:
        sample = x[idx]
        if sample.ndim == 3:
            _, h, w = sample.shape
            patch = max(4, int(min(h, w) * 0.2))
            top = max(0, (h // 2) - (patch // 2))
            left = max(0, (w // 2) - (patch // 2))
            sample[:, top : top + patch, left : left + patch] = rng.normal(2.2, 0.35, size=(sample.shape[0], patch, patch)).astype(np.float32)
        else:
            h, w = sample.shape
            patch = max(4, int(min(h, w) * 0.2))
            top = max(0, (h // 2) - (patch // 2))
            left = max(0, (w // 2) - (patch // 2))
            sample[top : top + patch, left : left + patch] = rng.normal(2.2, 0.35, size=(patch, patch)).astype(np.float32)
        x[idx] = sample

    classes = np.unique(y)
    label_flip_map: List[Dict[str, int]] = []
    for idx in label_indices:
        original = int(y[idx])
        candidates = [int(v) for v in classes.tolist() if int(v) != original]
        flipped = int(rng.choice(np.asarray(candidates, dtype=np.int64))) if candidates else int((original + 1) % 10)
        y[idx] = flipped
        label_flip_map.append({"index": int(idx), "from": original, "to": flipped})

    for idx in spike_indices:
        flat = x[idx].reshape(-1)
        spike_count = max(6, int(len(flat) * 0.03))
        spike_positions = rng.choice(len(flat), size=spike_count, replace=False)
        flat[spike_positions] = (flat[spike_positions] * 4.0) + 3.5
        x[idx] = flat.reshape(x[idx].shape)

    metadata = {
        "total_samples": total,
        "clean_samples": total - anomaly_count,
        "anomalous_samples": anomaly_count,
        "anomaly_type": ["pixel_level", "label_flip", "feature_spike"],
        "anomaly_percentage": round((anomaly_count / total) * 100.0, 2),
        "anomaly_indices": [int(i) for i in anomaly_indices],
        "anomaly_breakdown": {
            "pixel_level_indices": [int(i) for i in pixel_indices],
            "label_flip_indices": [int(i) for i in label_indices],
            "feature_spike_indices": [int(i) for i in spike_indices],
            "label_flip_map": label_flip_map,
        },
    }
    return x.astype(np.float32), y.astype(np.int64), metadata


def generate_golden_dataset() -> None:
    """Generate golden validation dataset with controlled anomalies."""
    os.makedirs("./data", exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

    golden_data = []
    golden_labels = []
    for i in range(min(100, len(dataset))):
        image, label = dataset[i]
        golden_data.append(image.numpy())
        golden_labels.append(label)

    golden_set = np.array(golden_data, dtype=np.float32)
    golden_labels = np.array(golden_labels, dtype=np.int64)

    golden_set, golden_labels, metadata = _inject_controlled_anomalies(golden_set, golden_labels, seed=42)

    np.save("./data/golden_set.npy", golden_set)
    np.save("./data/golden_labels.npy", golden_labels)
    with open("./data/dataset_metadata.json", "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)

    print(f"Golden dataset created: {golden_set.shape}")
    print(f"Labels shape: {golden_labels.shape}")
    print(f"Injected anomalies: {metadata['anomalous_samples']}/{metadata['total_samples']} ({metadata['anomaly_percentage']}%)")


def generate_hospital_data() -> None:
    """Generate placeholder data for simulated hospitals."""
    os.makedirs("./data/train_data", exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    full_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)

    total_size = len(full_dataset)
    split_size = total_size // 3

    for hospital_id in range(3):
        start_idx = hospital_id * split_size
        end_idx = (hospital_id + 1) * split_size if hospital_id < 2 else total_size

        hospital_data = []
        hospital_labels = []

        for i in range(start_idx, end_idx):
            image, label = full_dataset[i]
            hospital_data.append(image.numpy())
            hospital_labels.append(label)

        os.makedirs(f"./data/train_data/hospital_{hospital_id + 1}", exist_ok=True)
        np.save(f"./data/train_data/hospital_{hospital_id + 1}/data.npy", np.array(hospital_data, dtype=np.float32))
        np.save(f"./data/train_data/hospital_{hospital_id + 1}/labels.npy", np.array(hospital_labels, dtype=np.int64))

        print(f"Hospital {hospital_id + 1} data created: {len(hospital_data)} samples")


if __name__ == "__main__":
    print("Generating golden dataset with controlled anomalies...")
    generate_golden_dataset()

    print("Generating hospital training data...")
    generate_hospital_data()

    print("Data generation complete!")
