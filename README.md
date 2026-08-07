---
title: AetherEdge Vehicle Analytics
emoji: 🚗
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
app_port: 7860
---

# AetherEdge Vehicle Counting & Flow Analysis

Real-time computer vision system that detects vehicles in fixed-camera footage, tracks them across frames, and counts unique line crossings.

## Features
- SSDLite MobileNetV3 vehicle detection
- SORT Kalman filter tracking
- Unique line-crossing counting (no duplicates)
- Live annotated video stream
- CSV / JSON export
- SSE real-time alerts
