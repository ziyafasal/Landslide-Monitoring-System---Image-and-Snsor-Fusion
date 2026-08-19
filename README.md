**🌍 Landslide Monitoring and Prediction System**

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

 <img width="1342" height="690" alt="Screenshot 2026-08-19 161001" src="https://github.com/user-attachments/assets/a1f7c77f-72cb-4933-915b-92efc8161463" />
            

---

🔧 Hardware Components

- ESP32                      
- ESP32-CAM                    
- FC-28 Soil Moisture Sensor
- MPU6050
- MH-RD Rain Sensor            
- NEO-6M GPS                   
- Power Supply                 

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
- OpenCV
- scikit-image
- NumPy
- scikit-learn

Backend

- FastAPI
- Uvicorn
- SQLite
- WebSocket

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

<img width="1920" height="1200" alt="Screenshot 2026-03-26 201156" src="https://github.com/user-attachments/assets/1abce267-f5f4-4597-b25d-93ad3cec4f1e" />


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

Technologies: ESP32, ESP32-CAM, Python, PyTorch, FastAPI, SQLite

