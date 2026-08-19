"""
fusion_model.py
────────────────────────────────────────────────────────────────────
Multi-modal risk prediction model combining:
  - LSTM  → processes last N sensor readings (rain, moisture, vibration)
  - Image features → crack skeleton pixels, length, width, confidence
  - Fusion layer → weighted combination with tier-based priority rules

Risk classes:
  0 = SAFE      1 = WATCH      2 = WARNING      3 = CRITICAL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

RISK_LABELS = ["SAFE", "WATCH", "WARNING", "CRITICAL"]
RISK_DESCRIPTIONS = {
    "SAFE"    : "Slope is stable — normal monitoring",
    "WATCH"   : "Early signs detected — increase monitoring frequency",
    "WARNING" : "Multiple risk factors active — inspect site immediately",
    "CRITICAL": "Imminent failure risk — evacuate and alert",
}

# Sensor input features (order matters — must match your ESP32 data)
SENSOR_FEATURES = ["rain", "moisture", "vibration"]  # 3 features
SENSOR_SEQ_LEN  = 10   # use last 10 readings (tune this)

# Image input features
# [has_crack, confidence, skeleton_pixels_norm, length_px_norm, width_px_norm]
IMAGE_FEATURES = 5


# ─────────────────────────────────────────────
# Tier 1 Hard Override
# (bypasses the model completely — instant CRITICAL)
# ─────────────────────────────────────────────

VIBRATION_CRITICAL_THRESHOLD  = 5.0   # g  — immediate CRITICAL
VIBRATION_WARNING_THRESHOLD   = 3.0   # g  — immediate WARNING
SKELETON_CRITICAL_THRESHOLD   = 500   # px — immediate CRITICAL
SKELETON_WARNING_THRESHOLD    = 200   # px — immediate WARNING


def tier1_override(vibration: float, skeleton_pixels: int):
    """
    Tier 1 hard rules — these bypass the neural network entirely.
    Returns (risk_label, description) or None if no override.
    """
    # Vibration is the most critical signal
    if vibration >= VIBRATION_CRITICAL_THRESHOLD:
        return "CRITICAL", f"VIBRATION ALERT: {vibration:.2f}g — subsurface movement detected"

    # Large crack is equally critical
    if skeleton_pixels >= SKELETON_CRITICAL_THRESHOLD:
        return "CRITICAL", f"CRACK ALERT: {skeleton_pixels}px skeleton — imminent slope failure risk"

    # Vibration warning
    if vibration >= VIBRATION_WARNING_THRESHOLD:
        return "WARNING", f"High vibration: {vibration:.2f}g — monitor closely"

    # Medium crack
    if skeleton_pixels >= SKELETON_WARNING_THRESHOLD:
        return "WARNING", f"Significant crack: {skeleton_pixels}px — inspect site"

    return None  # no override — let the model decide


# ─────────────────────────────────────────────
# LSTM Sensor Branch
# ─────────────────────────────────────────────

class SensorLSTM(nn.Module):
    """
    Processes a sequence of sensor readings over time.
    Input:  (batch, seq_len, 3)  → [rain, moisture, vibration] per timestep
    Output: (batch, hidden_size) → learned sensor context vector
    """
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0,
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, (h_n, _) = self.lstm(x)
        # Take the last hidden state from the top layer
        last = h_n[-1]              # (batch, hidden_size)
        return self.norm(last)


# ─────────────────────────────────────────────
# Image Feature Branch
# ─────────────────────────────────────────────

class ImageFeatureMLP(nn.Module):
    """
    Processes crack image features (already extracted by FPN).
    Input:  (batch, 5) → [has_crack, confidence, skel_norm, len_norm, wid_norm]
    Output: (batch, 32)
    """
    def __init__(self, input_size=IMAGE_FEATURES, hidden_size=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, hidden_size),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# Fusion Model
# ─────────────────────────────────────────────

class LandslideRiskModel(nn.Module):
    """
    Fuses sensor LSTM output + image features → risk class (0-3).

    Priority weighting:
      - Vibration has highest weight in sensor branch
      - Crack features have highest weight in image branch
      - Fusion layer learns to combine both
    """
    def __init__(self,
                 sensor_hidden = 64,
                 image_hidden  = 32,
                 num_classes   = 4,    # SAFE, WATCH, WARNING, CRITICAL
                 dropout       = 0.3):
        super().__init__()

        self.sensor_branch = SensorLSTM(
            input_size  = len(SENSOR_FEATURES),
            hidden_size = sensor_hidden,
            num_layers  = 2,
            dropout     = dropout,
        )

        self.image_branch = ImageFeatureMLP(
            input_size  = IMAGE_FEATURES,
            hidden_size = image_hidden,
        )

        # Fusion MLP
        fused_size = sensor_hidden + image_hidden
        self.fusion = nn.Sequential(
            nn.Linear(fused_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, sensor_seq, image_features):
        """
        sensor_seq     : (batch, seq_len, 3)  — time-series sensor readings
        image_features : (batch, 5)           — extracted crack features
        Returns        : (batch, 4)           — logits for 4 risk classes
        """
        s = self.sensor_branch(sensor_seq)     # (batch, 64)
        i = self.image_branch(image_features)  # (batch, 32)
        fused = torch.cat([s, i], dim=1)       # (batch, 96)
        return self.fusion(fused)              # (batch, 4)


# ─────────────────────────────────────────────
# Input normalisation helpers
# ─────────────────────────────────────────────

# These max values are used to normalise inputs to [0, 1]
SENSOR_MAX = {
    "rain"      : 100.0,
    "moisture"  : 100.0,
    "vibration" : 7.0,
}
SKELETON_MAX = 1000.0   # normalise skeleton pixels
LENGTH_MAX   = 500.0    # normalise length px
WIDTH_MAX    = 100.0    # normalise width px


def normalise_sensor_reading(rain: float, moisture: float, vibration: float):
    """Normalise a single sensor reading to [0,1]."""
    return [
        min(rain       / SENSOR_MAX["rain"],       1.0),
        min(moisture   / SENSOR_MAX["moisture"],   1.0),
        min(vibration  / SENSOR_MAX["vibration"],  1.0),
    ]


def normalise_image_features(has_crack: bool, confidence: float,
                              skeleton_pixels: int,
                              length_px: float, width_px: float):
    """Normalise image features to [0,1]."""
    return [
        float(has_crack),
        float(confidence),
        min(skeleton_pixels / SKELETON_MAX, 1.0),
        min(length_px       / LENGTH_MAX,   1.0),
        min(width_px        / WIDTH_MAX,    1.0),
    ]


# ─────────────────────────────────────────────
# Predictor (wraps the model for easy use)
# ─────────────────────────────────────────────

class RiskPredictor:
    """
    High-level interface for risk prediction.
    Handles:
      - Tier 1 hard overrides (skip model)
      - Input normalisation
      - Model inference
      - Confidence score
    """

    def __init__(self, model_path: str = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model  = LandslideRiskModel().to(self.device)

        if model_path and __import__("os").path.exists(model_path):
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device))
            print(f"[RiskModel] Loaded from {model_path}")
        else:
            print("[RiskModel] No weights found — using untrained model")
            print("[RiskModel] Train with: python train_fusion.py")

        self.model.eval()
        self.sensor_buffer = []   # rolling buffer of last N readings

    def add_sensor_reading(self, rain: float, moisture: float, vibration: float):
        """Call this every time you receive a new sensor reading from ESP32."""
        norm = normalise_sensor_reading(rain, moisture, vibration)
        self.sensor_buffer.append(norm)
        # Keep only last SENSOR_SEQ_LEN readings
        if len(self.sensor_buffer) > SENSOR_SEQ_LEN:
            self.sensor_buffer.pop(0)

    def predict(self,
                rain: float, moisture: float, vibration: float,
                has_crack: bool, confidence: float,
                skeleton_pixels: int, length_px: float, width_px: float
                ) -> dict:
        """
        Full prediction using latest sensor reading + crack image features.

        Returns a dict with:
          risk_level, risk_description, confidence_score,
          method (model / tier1_override / rule_based),
          sensor_context, crack_context
        """
        # ── Step 1: Add current reading to buffer ─────────────────
        self.add_sensor_reading(rain, moisture, vibration)

        # ── Step 2: Tier 1 hard override ──────────────────────────
        override = tier1_override(vibration, skeleton_pixels)
        if override:
            level, desc = override
            return {
                "risk_level"      : level,
                "risk_description": desc,
                "confidence_score": 1.0,
                "method"          : "tier1_override",
                "sensor_context"  : self._sensor_context(rain, moisture, vibration),
                "crack_context"   : self._crack_context(has_crack, skeleton_pixels,
                                                         length_px, width_px),
            }

        # ── Step 3: Build sensor sequence tensor ──────────────────
        buffer = self.sensor_buffer.copy()

        # Pad with zeros if we don't have enough readings yet
        while len(buffer) < SENSOR_SEQ_LEN:
            buffer.insert(0, [0.0, 0.0, 0.0])

        sensor_tensor = torch.tensor([buffer], dtype=torch.float32).to(self.device)
        # shape: (1, seq_len, 3)

        # ── Step 4: Build image feature tensor ────────────────────
        img_feat = normalise_image_features(
            has_crack, confidence, skeleton_pixels, length_px, width_px)
        img_tensor = torch.tensor([img_feat], dtype=torch.float32).to(self.device)
        # shape: (1, 5)

        # ── Step 5: Model inference ────────────────────────────────
        with torch.no_grad():
            logits = self.model(sensor_tensor, img_tensor)
            probs  = F.softmax(logits, dim=1).squeeze()

        risk_idx   = int(probs.argmax())
        risk_level = RISK_LABELS[risk_idx]
        conf_score = float(probs[risk_idx])

        # ── Step 6: Moisture priming ───────────────────────────────
        # Tier 2: if soil is heavily saturated, upgrade WATCH → WARNING
        if risk_level == "WATCH" and moisture >= 80:
            risk_level = "WARNING"
            desc = (f"High soil saturation ({moisture:.0f}%) amplifies risk — "
                    f"{RISK_DESCRIPTIONS['WARNING']}")
        else:
            desc = RISK_DESCRIPTIONS[risk_level]

        return {
            "risk_level"      : risk_level,
            "risk_description": desc,
            "confidence_score": round(conf_score, 3),
            "method"          : "fusion_model",
            "probabilities"   : {
                RISK_LABELS[i]: round(float(probs[i]), 3)
                for i in range(4)
            },
            "sensor_context"  : self._sensor_context(rain, moisture, vibration),
            "crack_context"   : self._crack_context(has_crack, skeleton_pixels,
                                                     length_px, width_px),
        }

    def _sensor_context(self, rain, moisture, vibration):
        return {
            "rain"      : rain,
            "moisture"  : moisture,
            "vibration" : vibration,
            "readings_in_buffer": len(self.sensor_buffer),
        }

    def _crack_context(self, has_crack, skeleton_pixels, length_px, width_px):
        return {
            "has_crack"      : has_crack,
            "skeleton_pixels": skeleton_pixels,
            "length_px"      : length_px,
            "width_px"       : width_px,
        }
