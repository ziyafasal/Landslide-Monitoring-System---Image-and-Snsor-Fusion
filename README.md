Landslide Monitoring and Prediction System

Sensor Fusion + Image Fusion Based Landslide Monitoring and Risk Prediction

An IoT and machine-learning-based system for real-time landslide monitoring and risk prediction using environmental sensors, GPS, image processing, and deep learning.

The system combines sensor data and visual information to identify potential landslide conditions and provide a risk prediction through a real-time monitoring dashboard.

---

📌 Project Overview

Landslides can occur due to factors such as heavy rainfall, increased soil moisture, ground vibration, slope movement, and surface cracks.

This project proposes a multi-modal monitoring system that combines:

- 🌧️ Rainfall measurements
- 🌱 Soil moisture measurements
- 📐 Tilt and vibration measurements
- 📍 GPS location
- 📷 Camera-based crack detection and segmentation
- 🤖 Machine learning-based risk prediction
- 📊 Real-time web monitoring

The system collects sensor readings using an ESP32, captures images using an ESP32-CAM, processes visual information using deep-learning models, and combines the available information to estimate landslide risk.

---

🎯 Objectives

- Monitor environmental conditions associated with landslides.
- Collect real-time sensor data using ESP32.
- Detect cracks and potentially unstable regions from images.
- Combine sensor and image information for improved risk assessment.
- Predict landslide risk using machine learning.
- Display sensor readings, predictions, and GPS information through a web dashboard.
- Provide a scalable architecture for real-time landslide monitoring.

---

🏗️ System Architecture

                    ┌─────────────────────┐
                    │      Sensors        │
                    │                     │
                    │ • Soil Moisture     │
                    │ • Rainfall          │
                    │ • MPU6050            │
                    │ • GPS               │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       ESP32         │
                    │ Sensor Data         │
                    │ Acquisition         │
                    └──────────┬──────────┘
                               │
                               │
                               ▼
                    ┌─────────────────────┐
                    │     ESP32-CAM       │
                    │   Image Capture     │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │      Image Processing / ML     │
              │                                │
              │ • ResNet18                     │
              │ • MobileNetV2                  │
              │ • FPN Segmentation             │
              └────────────────┬───────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Sensor + Image    │
                    │       Fusion         │
                    │                      │
                    │       LSTM           │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Risk Prediction   │
                    │                     │
                    │  Low / Medium /     │
                    │       High          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   FastAPI Backend   │
                    │                     │
                    │ • Data Processing   │
                    │ • Database          │
                    │ • WebSocket         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Web Dashboard     │
                    │                     │
                    │ • Live Sensors      │
                    │ • Risk Prediction   │
                    │ • GPS Location      │
                    │ • Graphs             │
                    └─────────────────────┘

---

🔧 Hardware Components

Component| Purpose
ESP32| Main microcontroller and sensor data acquisition
ESP32-CAM| Captures images for visual analysis
FC-28 Soil Moisture Sensor| Measures soil moisture
MPU6050| Measures acceleration, tilt and vibration
MH-RD Rain Sensor| Detects rainfall
NEO-6M GPS| Provides geographical location
Power Supply| Powers the monitoring system

---

🤖 Machine Learning Pipeline

The project uses multiple deep-learning models for visual analysis and sensor-based prediction.

1. Image Classification

A MobileNetV2-based CNN is used for image classification and crack-related visual analysis.

MobileNetV2 was selected because of its relatively lightweight architecture, making it suitable for applications where computational efficiency is important.

---

2. Visual Stability Classification

A ResNet18 model is used as a primary visual filter to classify images based on visual stability and potential landslide-prone conditions.

---

3. Crack Segmentation

A Feature Pyramid Network (FPN) is used for pixel-level crack segmentation.

Configuration:

- Encoder: ResNet34
- Framework: "segmentation_models_pytorch"
- Image size: "256 × 256"

The segmentation model identifies the regions corresponding to cracks in the captured image.

---

4. Sensor Fusion and Risk Prediction

Sensor measurements are combined using an LSTM-based fusion model.

The model processes temporal sensor information such as:

- Soil moisture
- Rainfall
- Vibration
- Ground movement / tilt
- Visual information

The fused information is used to estimate the current landslide risk level.

---

📊 Risk Prediction

The system categorizes the estimated risk into different levels:

Sensor Data
     +
Image Analysis
     ↓
Fusion Model
     ↓
Risk Assessment
     ↓
┌──────────────┐
│     LOW      │
├──────────────┤
│    MEDIUM    │
├──────────────┤
│     HIGH     │
└──────────────┘

---

💻 Software & Technologies

Embedded System

- ESP32
- ESP32-CAM
- Arduino IDE
- C/C++

Machine Learning

- Python
- PyTorch
- Torchvision
- segmentation_models_pytorch
- OpenCV
- scikit-image
- NumPy
- scikit-learn

Backend

- FastAPI
- Uvicorn
- SQLite
- WebSocket

Frontend

- HTML
- CSS
- JavaScript
- Chart.js

Development & Training

- Visual Studio Code
- Google Colab
- NVIDIA T4 GPU

---

📁 Project Structure

Landslide-Monitoring-System/
│
├── ML files/
│   ├── FPN/
│   │   └── best_model.pth
│   ├── ...
│   └── ...
│
├── Microcontroller/
│   ├── ESP32/
│   ├── ESP32-CAM/
│   └── ...
│
├── Monitoring/
│   ├── Backend/
│   ├── Frontend/
│   └── ...
│
├── .gitignore
└── README.md

«The exact folder contents may vary depending on the current implementation.»

---

🔄 System Workflow

Step 1 — Data Acquisition

The ESP32 collects environmental information from the connected sensors.

Step 2 — Image Acquisition

The ESP32-CAM captures images of the monitored area.

Step 3 — Image Processing

Captured images are processed using computer vision and deep-learning models.

Step 4 — Crack Detection and Segmentation

The trained CNN and FPN models analyze the images and identify cracks or potentially unstable visual regions.

Step 5 — Sensor Fusion

Sensor measurements are combined with the visual analysis results.

Step 6 — Risk Prediction

The LSTM-based fusion model estimates the landslide risk level.

Step 7 — Backend Processing

The FastAPI server receives and processes the data and maintains the required system information.

Step 8 — Real-Time Dashboard

The monitoring dashboard displays:

- Sensor readings
- Real-time graphs
- Risk prediction
- GPS location
- Visual prediction results

---

📍 GPS Monitoring

The NEO-6M GPS module provides the geographical coordinates of the monitoring system.

This allows the system to associate detected conditions with a specific monitoring location.

---

📈 Real-Time Monitoring

The web dashboard is designed to provide real-time visualization of system information.

The dashboard can display:

- 🌧️ Rainfall
- 🌱 Soil moisture
- 📐 Tilt / vibration
- 📍 GPS coordinates
- 📷 Image analysis results
- ⚠️ Landslide risk
- 📊 Sensor graphs

WebSocket communication is used to support real-time data updates.

---

🧠 Machine Learning Model Files

The repository contains trained model files used by the system.

Large model files such as ".pth" files may be managed separately using Git Large File Storage (Git LFS) when required.

---

🚀 Installation

Clone the repository

git clone https://github.com/YOUR-USERNAME/Landslide-Monitoring-System---Image-and-Sensor-Fusion.git

Move into the project directory:

cd Landslide-Monitoring-System---Image-and-Sensor-Fusion

---

Python Environment

Create a virtual environment:

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

Install the required Python packages:

pip install -r requirements.txt

---

▶️ Running the Backend

Navigate to the backend directory:

cd Monitoring

Run the FastAPI server:

uvicorn main:app --reload

The backend can then be accessed locally through the configured server address.

---

🔌 Hardware Setup

Connect the sensors to the ESP32 according to the project's hardware configuration.

Main components include:

FC-28 Soil Moisture Sensor → ESP32
MH-RD Rain Sensor          → ESP32
MPU6050                    → ESP32 (I²C)
NEO-6M GPS                 → ESP32
ESP32-CAM                  → Image Acquisition

The exact GPIO configuration can be found in the corresponding Arduino source files.

---

📷 Project Screenshots

Add screenshots of your project here.

Recommended screenshots:

1. Hardware prototype
2. ESP32-CAM setup
3. Web dashboard
4. Sensor graphs
5. Crack detection result
6. Crack segmentation result
7. Landslide risk prediction
8. GPS location

Example:

![Project Hardware](docs/hardware.jpg)

![Monitoring Dashboard](docs/dashboard.png)

![Crack Segmentation](docs/crack-segmentation.png)

---

🎥 Demo

Add your project demonstration video here if available.

Demo: Coming soon

---

📌 Key Features

- ✅ Real-time environmental monitoring
- ✅ ESP32-based IoT architecture
- ✅ ESP32-CAM image acquisition
- ✅ Soil moisture monitoring
- ✅ Rainfall monitoring
- ✅ Vibration and tilt monitoring
- ✅ GPS-based location tracking
- ✅ CNN-based image analysis
- ✅ Crack segmentation using FPN
- ✅ LSTM-based sensor fusion
- ✅ Landslide risk prediction
- ✅ FastAPI backend
- ✅ WebSocket-based real-time communication
- ✅ Interactive monitoring dashboard

---

🔮 Future Improvements

Possible future improvements include:

- Integration of additional environmental sensors
- Improved multi-modal fusion techniques
- Edge deployment of lightweight ML models
- Cloud-based monitoring
- SMS / email / mobile notifications
- Historical data analytics
- Improved geospatial visualization
- Larger and more diverse training datasets
- Deployment across multiple monitoring locations
- Automated early-warning mechanisms

---

👩‍💻 Project

Project: Landslide Monitoring and Prediction: Sensor and Image Fusion

Domain: Electronics and Communication Engineering / IoT / Machine Learning / Computer Vision

Technologies: ESP32, ESP32-CAM, Python, PyTorch, FastAPI, WebSocket, SQLite, JavaScript

