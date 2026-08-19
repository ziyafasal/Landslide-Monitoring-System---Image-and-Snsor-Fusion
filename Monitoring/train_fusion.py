
"""
train_fusion.py (Updated for Sensor-Only Resilience)
────────────────────────────────────────────────────────────────────
Trains the LandslideRiskModel using synthetic data.
Now includes 'Missing Image' simulation so the model can predict 
using only sensor data when the camera is offline.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# Assuming these are defined in your fusion_model.py
from fusion_model import (
    LandslideRiskModel, SENSOR_SEQ_LEN, RISK_LABELS,
    normalise_sensor_reading, normalise_image_features,
)

# ─────────────────────────────────────────────
# Synthetic data generator
# ─────────────────────────────────────────────

def generate_sample(risk_class: int, noise: float = 0.05, missing_img_prob: float = 0.25):
    """
    Generate one (sensor_sequence, image_features, label) sample.
    - missing_img_prob: Chance that image features are zeroed out to 
      train the model to rely on sensors only.
    """
    rng = np.random

    # 1. Generate Sensor Base Values
    if risk_class == 0:   # SAFE
        rain, moisture, vibration = rng.uniform(0, 20), rng.uniform(10, 50), rng.uniform(0, 1.0)
        has_crack, skel, conf = False, 0, rng.uniform(0.1, 0.4)
    elif risk_class == 1:  # WATCH
        rain, moisture, vibration = rng.uniform(20, 50), rng.uniform(40, 70), rng.uniform(0.5, 2.0)
        has_crack = rng.random() > 0.4
        skel, conf = (rng.uniform(30, 100), rng.uniform(0.4, 0.7)) if has_crack else (0, 0.1)
    elif risk_class == 2:  # WARNING
        rain, moisture, vibration = rng.uniform(40, 80), rng.uniform(60, 85), rng.uniform(1.5, 3.5)
        has_crack = rng.random() > 0.2
        skel, conf = (rng.uniform(100, 300), rng.uniform(0.6, 0.9)) if has_crack else (0, 0.3)
    else:                  # CRITICAL
        rain, moisture, vibration = rng.uniform(60, 100), rng.uniform(75, 100), rng.uniform(3.0, 7.0)
        has_crack = rng.random() > 0.1
        skel, conf = (rng.uniform(300, 1000), rng.uniform(0.8, 1.0)) if has_crack else (0, 0.5)

    # 2. Build sensor sequence (LSTM input)
    sequence = []
    for t in range(SENSOR_SEQ_LEN):
        r = np.clip(rain + rng.normal(0, noise * 100), 0, 100)
        m = np.clip(moisture + rng.normal(0, noise * 100), 0, 100)
        v = np.clip(vibration + rng.normal(0, noise * 7), 0, 7)
        sequence.append(normalise_sensor_reading(r, m, v))

    # 3. Image features (Vision input)
    length_px = skel * rng.uniform(1.0, 2.5)
    width_px  = skel * rng.uniform(0.02, 0.15)
    
    # --- CRITICAL UPDATE: Simulate Sensor-Only Mode ---
    if rng.random() < missing_img_prob:
        # Pass zeros for everything to simulate "No Image Data"
        img_feat = [0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        img_feat = normalise_image_features(has_crack, conf, int(skel), length_px, width_px)

    return sequence, img_feat, risk_class

# ─────────────────────────────────────────────
# Dataset Class
# ─────────────────────────────────────────────

class LandslideDataset(Dataset):
    def __init__(self, n_samples=5000):
        self.samples = []
        per_class = n_samples // 4
        for cls in range(4):
            for _ in range(per_class):
                seq, img, label = generate_sample(cls)
                self.samples.append((
                    torch.tensor(seq, dtype=torch.float32),
                    torch.tensor(img, dtype=torch.float32),
                    torch.tensor(label, dtype=torch.long),
                ))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

def train(n_samples=6000, epochs=80, batch_size=64, lr=1e-3, save_dir="checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    dataset = LandslideDataset(n_samples)
    n_val = int(len(dataset) * 0.2)
    trn_ds, val_ds = random_split(dataset, [len(dataset)-n_val, n_val])
    
    trn_loader = DataLoader(trn_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False)

    model = LandslideRiskModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    print(f"Starting Training on {device}...")
    best_acc = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for sensor, img, labels in trn_loader:
            sensor, img, labels = sensor.to(device), img.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sensor, img)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for sensor, img, labels in val_loader:
                outputs = model(sensor.to(device), img.to(device))
                preds = outputs.argmax(1).cpu()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        acc = correct / total
        scheduler.step(total_loss)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{save_dir}/fusion_model.pth")
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {total_loss/len(trn_loader):.4f} | Val Acc: {acc:.3f}")

    print(f"\nTraining Complete. Best Accuracy: {best_acc:.3f}")

if __name__ == "__main__":
    train()