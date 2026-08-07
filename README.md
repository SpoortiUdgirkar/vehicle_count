# 🚗 AetherEdge — Vehicle Counting & Flow Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-black?style=for-the-badge&logo=flask&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)

**A real-time computer vision system that detects, tracks, and counts vehicles crossing a virtual line in fixed-camera footage.**

</div>

---

## 📌 Overview

AetherEdge is a full-stack vehicle counting solution built for the **Vehicle Counting & Flow Analysis Challenge**. It uses deep learning detection combined with multi-object tracking to accurately count unique vehicle crossings — without double-counting — and presents results through a live web dashboard.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🔍 **Vehicle Detection** | SSDLite MobileNetV3-Large (COCO pretrained) |
| 🎯 **Multi-Object Tracking** | SORT — Kalman Filter + Hungarian Algorithm |
| ✂️ **Duplicate Prevention** | NMS (IoU=0.40) + Spatiotemporal cooldown (130px / 1.8s) |
| 📹 **Live Video Stream** | MJPEG stream at 15fps in browser |
| 📊 **Real-time Dashboard** | KPIs, event log, alerts, progress bar |
| 📁 **Export Options** | CSV log, JSON summary, annotated MP4 video |
| 🐳 **Deployment Ready** | Dockerized Flask application |
| 🌐 **Public Sharing** | ngrok tunnel support via `launch.py` |

---

## 🏗️ System Architecture

```
Video Input (cars.mp4)
        │
        ▼
┌─────────────────────────────────────┐
│  Frame Extractor (OpenCV)           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Vehicle Detector (detector.py)     │
│  SSDLite MobileNetV3 @ 320×320      │
│  Confidence threshold: 0.42         │
│  NMS IoU threshold: 0.40            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  SORT Tracker (tracker.py)          │
│  Kalman Filter + Hungarian Matching │
│  min_hits=3, max_age=30             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Line Counter (counter.py)          │
│  Cross-product crossing detection   │
│  Spatiotemporal duplicate guard     │
└──────────┬───────────────┬──────────┘
           │               │
           ▼               ▼
      MJPEG Stream    Annotated MP4
      (Live Feed)     + CSV + JSON
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Web Framework | Flask |
| Detection Model | SSDLite MobileNetV3-Large |
| Deep Learning | PyTorch + TorchVision |
| Computer Vision | OpenCV |
| Tracking | SORT (Kalman Filter + Hungarian Algorithm) |
| Frontend | HTML5 · CSS3 · Vanilla JavaScript |
| Real-time Events | Server-Sent Events (SSE) |
| Video Streaming | MJPEG Multipart Stream |
| Deployment | Docker |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/SpoortiUdgirkar/vehicle_count.git
cd vehicle_count
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
python vehicle_app.py
```

### 4. Open in browser
```
http://localhost:5000
```

---

## 🌐 Run with Public URL (ngrok)

Share the dashboard with anyone using a public link:

```bash
python launch.py --authtoken YOUR_NGROK_TOKEN
```

Get a free token at: https://dashboard.ngrok.com/get-started/your-authtoken

Your link will look like:
```
Public: https://xxxx-xx-xx-xxx.ngrok-free.app
```

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t vehicle-counter .

# Run container
docker run -p 5000:5000 vehicle-counter
```

---

## 📊 How It Works

### Detection
- **Model:** SSDLite MobileNetV3-Large pretrained on MS-COCO
- **Classes detected:** car (3), bus (6), truck (8), motorcycle (4)
- **Input resolution:** 320×320 (upscaled bounding boxes mapped back to original)
- **Confidence threshold:** 0.42 (filters weak/false detections)
- **NMS:** IoU=0.40 (removes duplicate boxes on same vehicle)

### Tracking
- **Kalman Filter** predicts next bounding box position between frames
- **Hungarian Algorithm** optimally matches predictions to new detections
- A track needs **3 consecutive hits** to be confirmed (avoids ghost tracks)
- Tracks persist for **30 frames** without a detection before being removed

### Counting
- A virtual counting line is defined by two configurable points (default: P1=220,420 → P2=1080,420)
- Crossing is detected by a **sign change in cross-product** of centroid vs. line vector
- **Spatiotemporal cooldown:** After counting, any crossing within **130px radius** and **1.8 seconds** is ignored — prevents double-counting after ID switches

---

## 📁 Project Structure

```
vehicle_count/
├── vehicle_app.py          # Flask server + processing loop + all API routes
├── detector.py             # SSDLite inference + NMS
├── tracker.py              # SORT: Kalman filter + Hungarian matching
├── counter.py              # Line crossing detection + cooldown logic
├── launch.py               # App launcher with ngrok tunnel
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container definition
├── Procfile                # For Railway/Render deployment
├── runtime.txt             # Python version pin
├── templates/
│   └── index.html          # Dashboard HTML
├── static/
│   ├── css/style.css       # Dark theme styling
│   └── js/app.js           # Real-time UI controller
└── datasets/
    └── video/
        └── cars.mp4        # Challenge test video
```

---

## 🖥️ Dashboard Features

- **Live MJPEG stream** — annotated video with bounding boxes and tracking trails
- **KPI cards** — Total Vehicles, Cars, Heavy Vehicles, Processing FPS
- **Crossing Events log** — timestamp, Track ID, class, direction per crossing
- **Alert banners** — real-time pop-up on each vehicle crossing
- **Configurable line** — adjust P1/P2 coordinates from the UI
- **Speed modes** — Max Accuracy / Fast (skip 1 frame) / Ultra-fast (skip 2 frames)
- **Results modal** — auto-appears at end of video with download links
- **Export** — CSV event log, JSON summary, annotated MP4

---

## 📤 Outputs

| Output | Format | Description |
|---|---|---|
| Total count | Integer | Unique vehicles that crossed the line |
| Annotated video | MP4 | Bounding boxes, track IDs, trails, HUD |
| Event log | CSV | Per-crossing: timestamp, ID, class, direction |
| Summary | JSON | Aggregate counts by class and direction |
| Live stream | MJPEG | Real-time annotated feed in dashboard |

---

## 🔄 Application Lifecycle

```
READY ──[Start Analysis]──► RUNNING ──[video ends]──► FINISHED
  ▲                                                       │
  └────────────────[Reset / Run Again]────────────────────┘
```

- **READY** — Model loaded, waiting for user input
- **RUNNING** — Processing active, live stream on, Stop button visible  
- **FINISHED** — Results modal auto-appears with full count and download options

---

## 📋 Requirements

```
flask
opencv-python-headless
torch
torchvision
filterpy
scipy
pyngrok
```

---

## 👩‍💻 Author

**Spoorthi Udgirkar**  
Computer Vision Challenge Submission — AetherEdge Vehicle Analytics System
