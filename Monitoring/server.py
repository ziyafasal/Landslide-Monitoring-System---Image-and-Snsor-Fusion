"""
main.py — Landslide Monitor Server
Integrates: ResNet18 Gatekeeper + FPN crack segmentation + LSTM sensor fusion
"""

import json
import math
import os
import io
import sqlite3
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import models, transforms

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False

try:
    from skimage.morphology import skeletonize
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

from fusion_model import RiskPredictor

# ─────────────────────────────────────────
#  CONFIG & DEVICE
# ─────────────────────────────────────────
MODEL_DIR         = "checkpoints"
FUSION_MODEL_PATH = os.path.join(MODEL_DIR, "fusion_model.pth")
DB_PATH           = "monitoring.db"
device            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image Normalization Constants
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────
#  DATABASE
# ─────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sensor_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, moisture REAL, rain REAL, vibration REAL,
        lat REAL, lon REAL, altitude REAL, satellites INTEGER
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, has_crack INTEGER, confidence REAL,
        skeleton_pixels INTEGER, length_px REAL, width_px REAL,
        rain REAL, moisture REAL, vibration REAL,
        risk_level TEXT, risk_description TEXT, method TEXT
    )''')
    conn.commit()
    conn.close()

def save_sensor(data: dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO sensor_data
        (timestamp,moisture,rain,vibration,lat,lon,altitude,satellites)
        VALUES (?,?,?,?,?,?,?,?)''', (
        datetime.now().isoformat(),
        data.get("moisture",0), data.get("rain",0), data.get("vibration",0),
        data.get("lat",0), data.get("lon",0),
        data.get("altitude",0), data.get("satellites",0),
    ))
    conn.commit()
    conn.close()

def save_full_result(result: dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO predictions
        (timestamp,has_crack,confidence,skeleton_pixels,length_px,width_px,
         rain,moisture,vibration,risk_level,risk_description,method)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''', (
        datetime.now().isoformat(),
        int(result.get("has_crack", False)),
        result.get("confidence",    0.0),
        result.get("skeleton_pixels", 0),
        result.get("length_px",     0.0),
        result.get("width_px",      0.0),
        result.get("rain",          0.0),
        result.get("moisture",      0.0),
        result.get("vibration",     0.0),
        result.get("risk_level",    "SAFE"),
        result.get("risk_description", ""),
        result.get("method",        "unknown"),
    ))
    conn.commit()
    conn.close()

# ─────────────────────────────────────────
#  RESNET18 CLASSIFIER (GATEKEEPER)
# ─────────────────────────────────────────
crack_classifier = None
classifier_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

def load_crack_classifier():
    global crack_classifier
    path = os.path.join(MODEL_DIR, "crack_recognition.pth")
    if not os.path.exists(path):
        print("[Classifier] Not found")
        return
    try:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval().to(device)
        crack_classifier = model
        print("[Classifier] ResNet18 Gatekeeper Loaded")
    except Exception as e:
        print(f"[Classifier] Error: {e}")

# ─────────────────────────────────────────
#  FPN SEGMENTATION MODEL
# ─────────────────────────────────────────
fpn_model = None
IMG_SIZE  = 256

def load_fpn_model():
    global fpn_model, IMG_SIZE
    if not SMP_AVAILABLE: return
    cfg_path = os.path.join(MODEL_DIR, "config.json")
    cfg = {"arch": "fpn", "encoder": "resnet34", "img_size": 256}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f: cfg.update(json.load(f))
    IMG_SIZE = cfg["img_size"]
    weights  = os.path.join(MODEL_DIR, "best_model.pth")
    if not os.path.exists(weights):
        print("[FPN] Weights not found"); return
    arch_map = {"fpn": smp.FPN, "linknet": smp.Linknet, "unet": smp.Unet, "deeplabv3": smp.DeepLabV3Plus}
    try:
        m = arch_map[cfg["arch"]](encoder_name=cfg["encoder"], encoder_weights=None, in_channels=3, classes=1)
        m.load_state_dict(torch.load(weights, map_location=device))
        m.eval().to(device)
        fpn_model = m
        print(f"[FPN] Loaded ({device})")
    except Exception as e:
        print(f"[FPN] Failed: {e}")

@torch.no_grad()
def run_fpn(img_bgr: np.ndarray, threshold=0.5):
    h, w = img_bgr.shape[:2]
    rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb  = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    rgb  = (rgb - np.array(MEAN)) / np.array(STD)
    t    = torch.from_numpy(rgb.transpose(2,0,1)).unsqueeze(0).float().to(device)
    prob = torch.sigmoid(fpn_model(t)).squeeze().cpu().numpy()
    mask = (prob > threshold).astype(np.uint8) * 255
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST), float(prob.max())

def measure_crack(mask: np.ndarray, grid_size=10):
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0 or not SKIMAGE_AVAILABLE:
        return {"skeleton_pixels": 0, "length_px": 0.0, "width_px": 0.0}
    skeleton    = skeletonize(binary.astype(bool)).astype(np.uint8)
    skel_pixels = int(skeleton.sum())
    pts         = np.column_stack(np.where(skeleton > 0))
    if len(pts) < 2:
        return {"skeleton_pixels": skel_pixels, "length_px": 0.0, "width_px": 0.0}
    h, w = mask.shape[:2]
    pts  = pts[np.lexsort((pts[:,1], pts[:,0]))]
    cell_map = {}
    for y, x in pts:
        cell = (int(y)//grid_size, int(x)//grid_size)
        cell_map.setdefault(cell, []).append((y, x))
    ordered = sorted([(int(np.mean([p[0] for p in v])), int(np.mean([p[1] for p in v])))
                      for v in cell_map.values()], key=lambda p: (p[0], p[1]))
    length_px = sum(math.hypot(ordered[i+1][1]-ordered[i][1], ordered[i+1][0]-ordered[i][0])
                    for i in range(len(ordered)-1))
    
    def seg_w(cy, cx, dy, dx):
        d = math.hypot(dx, dy)
        if d == 0: return 0
        py, px = -dx/d, dy/d
        def edge(s):
            for n in range(1, max(h, w)):
                ny, nx = int(round(cy+s*n*py)), int(round(cx+s*n*px))
                if not (0<=ny<h and 0<=nx<w): return None
                if binary[ny,nx]==0: return int(round(cy+s*(n-1)*py)), int(round(cx+s*(n-1)*px))
            return None
        t, b = edge(1), edge(-1)
        return math.hypot(t[0]-b[0], t[1]-b[1]) if t and b else 0

    widths = [seg_w((ordered[i][0]+ordered[i+1][0])//2, (ordered[i][1]+ordered[i+1][1])//2,
                    ordered[i+1][0]-ordered[i][0], ordered[i+1][1]-ordered[i][1])
              for i in range(len(ordered)-1)]
    widths = [x for x in widths if x > 0]
    return {"skeleton_pixels": skel_pixels, "length_px": round(length_px, 1),
            "width_px": round(float(np.mean(widths)) if widths else 0, 1)}

# ─────────────────────────────────────────
#  FUSION RISK MODEL
# ─────────────────────────────────────────
risk_predictor = None

def load_risk_model():
    global risk_predictor
    path = FUSION_MODEL_PATH if os.path.exists(FUSION_MODEL_PATH) else None
    risk_predictor = RiskPredictor(model_path=path, device=str(device))
    print(f"[Risk] Fusion model ready (path: {path})")

# ─────────────────────────────────────────
#  COMBINED ANALYSIS (GATEKEEPER LOGIC)
# ─────────────────────────────────────────
def analyse(image_bytes: bytes, sensor: dict) -> dict:
    rain      = float(sensor.get("rain",      0))
    moisture  = float(sensor.get("moisture",  0))
    vibration = float(sensor.get("vibration", 0))

    has_crack, confidence = False, 0.0
    skel, l, w = 0, 0.0, 0.0

    if image_bytes:
        try:
            img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # 1. Gatekeeper Classification
            if crack_classifier is not None:
                input_tensor = classifier_transform(img_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = crack_classifier(input_tensor)
                    prob = torch.softmax(output, dim=1)
                    conf_values, indices = torch.max(prob, 1)
                    is_crack_detected = (indices.item() == 1)
                    confidence = float(conf_values.item())

                # 2. Segmentation (Only if Classifier detects a crack)
                if is_crack_detected and fpn_model is not None:
                    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                    mask, _ = run_fpn(img_cv)
                    
                    if int((mask > 0).sum()) > 100:
                        has_crack = True
                        m    = measure_crack(mask)
                        skel = m["skeleton_pixels"]
                        l    = m["length_px"]
                        w    = m["width_px"]
                    else:
                        has_crack = False # FPN noise filter
                else:
                    has_crack = False
        except Exception as e:
            print(f"[Analyse] Error: {e}")

    # 3. Fusion Prediction
    risk = risk_predictor.predict(
        rain=rain, moisture=moisture, vibration=vibration,
        has_crack=has_crack, confidence=confidence,
        skeleton_pixels=skel, length_px=l, width_px=w,
    )

    return {
        "has_crack"       : has_crack,
        "confidence"      : round(confidence, 4),
        "skeleton_pixels" : skel,
        "length_px"       : l,
        "width_px"        : w,
        "rain"            : rain,
        "moisture"        : moisture,
        "vibration"       : vibration,
        "risk_level"      : risk["risk_level"],
        "risk_description": risk["risk_description"],
        "risk_confidence" : risk["confidence_score"],
        "method"          : risk["method"],
        "probabilities"   : risk.get("probabilities", {}),
    }

# ─────────────────────────────────────────
#  FASTAPI
# ─────────────────────────────────────────
app = FastAPI(title="Landslide Monitor")
app.mount("/static", StaticFiles(directory="."), name="static")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ws_clients: list[WebSocket] = []
latest_sensor: dict = {"rain": 0, "moisture": 0, "vibration": 0}

async def broadcast(data: dict):
    msg = json.dumps(data)
    for ws in ws_clients.copy():
        try: await ws.send_text(msg)
        except Exception: ws_clients.remove(ws)

@app.on_event("startup")
async def startup():
    init_db()
    load_crack_classifier()
    load_fpn_model()
    load_risk_model()
    print("[Server] Ready → http://0.0.0.0:8000")

@app.get("/dashboard")
async def dashboard(): return FileResponse("dashboard.html")

@app.post("/sensors")
async def receive_sensors(request: Request):
    global latest_sensor
    try:
        data = await request.json()
        save_sensor(data)
        latest_sensor = data
        risk_predictor.add_sensor_reading(data.get("rain", 0), data.get("moisture", 0), data.get("vibration", 0))
        data["type"], data["timestamp"] = "sensor", datetime.now().isoformat()
        await broadcast(data)
        return JSONResponse({"status": "ok"})
    except Exception as e: return JSONResponse({"status": "error", "message": str(e)}, status_code=400)

@app.post("/predict")
async def predict(request: Request):
    try:
        image_bytes = await request.body()
        result      = analyse(image_bytes, latest_sensor)
        save_full_result(result)
        result["type"], result["timestamp"] = "prediction", datetime.now().isoformat()
        await broadcast(result)
        return JSONResponse(result)
    except Exception as e: return JSONResponse({"status": "error", "message": str(e)}, status_code=400)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_clients.append(websocket)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect: ws_clients.remove(websocket)

@app.get("/history/sensors")
async def sensor_history():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT * FROM sensor_data ORDER BY id DESC LIMIT 50")
    rows, keys = c.fetchall(), ["id","timestamp","moisture","rain","vibration","lat","lon","altitude","satellites"]
    conn.close()
    return [dict(zip(keys, r)) for r in rows]

@app.get("/history/predictions")
async def prediction_history():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 20")
    rows, keys = c.fetchall(), ["id","timestamp","has_crack","confidence","skeleton_pixels","length_px","width_px","rain","moisture","vibration","risk_level","risk_description","method"]
    conn.close()
    return [dict(zip(keys, r)) for r in rows]

@app.get("/status")
async def status():
    return {
        "server": "running", "fpn_model": "loaded" if fpn_model else "not loaded",
        "classifier": "loaded" if crack_classifier else "not loaded",
        "fusion_model": "loaded" if os.path.exists(FUSION_MODEL_PATH) else "not trained",
        "device": str(device), "timestamp": datetime.now().isoformat(),
    }

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)