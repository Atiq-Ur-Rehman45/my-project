# 🛡️ AI-Based Face & Weapon Recognition System

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-orange.svg)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-YuNet_%2F_SFace-red.svg)

A professional-grade surveillance and security dashboard featuring state-of-the-art **YuNet** face detection and **SFace** deep learning recognition. Optimized for high-accuracy tracking at distance and real-time weapon detection.

---

## 🚀 Key Features

*   **Elite Accuracy Bundle**:
    *   **Automatic Outlier Filter**: PRunes poor-quality enrollment samples using pairwise cosine distance.
    *   **Top-K Mean Scoring**: Average of the top 3 similarity matches per person to prevent "one-off" false positives.
    *   **Weighted Temporal Consensus**: Exponential decay voting (`0.9^n`) for stable, flicker-free identity acquisition.
    *   **Adaptive Small-Face Sharpening**: Automatic Unsharp Mask (USM) filter with hysteresis for accurate recognition of distant subjects.
*   **Dual-Layer Security**:
    *   **Real-time Recognition**: Instant lookup against a local criminal/VIP database.
    *   **Weapon Detection**: Async YOLOv8 processing to detect firearms without impacting video FPS.
*   **Professional Dashboard**:
    *   High-performance MJPEG streaming with WebSocket HUD overlays.
    *   Automated incident logging and snapshot capture.
    *   Responsive web interface for system management and enrollment.

---

## 🏗️ Technical Architecture

| Layer | Component | Details |
| :--- | :--- | :--- |
| **Detection** | **YuNet** | CNN-based detector optimized for speed and occluded faces. |
| **Recognition** | **SFace** | Deep Learning 128D embeddings with sub-pixel alignment precision. |
| **Analysis** | **YOLOv8** | ONNX-accelerated object detection for weapon identification. |
| **Database** | **SQLite3** | Secure local storage for criminal records and event logs. |
| **UI** | **Flask + SocketIO** | Real-time dashboard with ultra-low latency alerts. |

---

## 📁 Project Structure

```bash
face_recognition_system/
├── web_app.py            # Main Entry Point (Launch Dashboard)
├── config.py             # Global Configuration (Thresholds, Paths, UI)
├── face_engine.py        # Core YuNet + SFace Engine
├── database.py           # Identity & Log Management (SQLite)
├── trainer.py            # Embedding Generation & Augmentation
├── legacy/               # Archived terminal-based interfaces
├── data/                 # Local records (Snapshots, Databases)
└── models/               # Pre-trained ONNX Model files
```

---

## ⚙️ Installation & Setup

### 1. Requirements
Ensure you have Python 3.11+ installed.
```bash
pip install -r requirements.txt
```

### 2. Launching the Dashboard
The system is fully web-native. To start the dashbaord:
```bash
python web_app.py
```
Then open: **[http://localhost:5000](http://localhost:5000)**

### 3. Enrollment Workflow
1.  Navigate to **Enroll New Person**.
2.  The system will guide you through a **Pose Challenge** (Front, Left, Right, Up, Down).
3.  Ensure "Training" is triggered to generate the optimized embedding matrix.

---

## 🎯 Elite Optimization Details

This project utilizes advanced computer vision techniques to push SFace to its theoretical limit:
- **Hysteresis Smoothing**: Prevents "blinking" labels when subjects are at the edge of the detection range.
- **Motion-Aware Hold**: Identity persistence increases when subjects are moving/walking to survive motion blur.
- **LAB-Space CLAHE**: Lighting normalization is performed in the Lab color space to maintain color integrity while maximizing contrast.




