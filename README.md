# 🚗 Parking Lot Occupancy Detection (YOLOv8) — Image & Video Inference System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-orange)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)](#)
[![License](https://img.shields.io/badge/License-MIT-green)](#)

A production-style **Computer Vision inference system** that detects and classifies parking slots from **images and videos** into:

- 🟩 **Free slots**
- 🟥 **Occupied slots**
- 🟨 **Partially free slots**

This project goes beyond model training — it provides a complete **ML delivery pipeline** including:
✅ Streamlit dashboard (upload → inference → annotated output → download)  
✅ CLI tool for image/video inference automation  
✅ Structured logging for reproducibility (`outputs/state.json`)  

---

## 🌐 Live Demo (Deployed on Streamlit Cloud)

✅ **Try it here:**  
**https://parking-lot-detection-v-2-jzd4kygxdn7qtj9eua9lr9.streamlit.app/**

> This deployment demonstrates real-world AI productization: model inference exposed as a usable application.

---

## 📌 Executive Summary

**Problem:** Manual monitoring of parking availability is slow and inaccurate, especially for large parking lots.  
**Solution:** Use a YOLOv8-based detector trained on a custom dataset to classify parking slot occupancy from visual input.  
**Result:** A complete inference system that outputs:
- Total counts (Free / Occupied / Partial)
- Annotated media (PNG / MP4)
- JSON log for auditing and reproducibility

---

## ✅ Key Capabilities

### 🧠 ML / Vision
- Custom-trained **YOLOv8 detector**
- Multi-class classification (Free / Occupied / Partial)
- Confidence-based filtering to reduce false positives
- CPU-supported inference (suitable for cloud hosting)

### ⚙️ Engineering / Productization
- **Streamlit App** for media upload, visualization, and downloads
- **Video inference** frame-by-frame processing
- Color-coded bounding boxes + overlay counts
- Structured state tracking: `outputs/state.json`
- Clean modular repo structure (CLI + UI)

---

## 🏗️ System Architecture

### High-Level Pipeline


### What the model returns (YOLO Output)
- `xyxy` bounding box coordinates  
- `conf` confidence score  
- `cls` predicted class id  

The system uses confidence to filter noise and uses class labels to compute slot counts.

---

## 📁 Repository Structure

parking-lot-detection-v-2/
│
├── dashboard/
│ └── app.py # Streamlit dashboard (upload + inference + download)
│
├── src/
│ ├── run_app.py # CLI inference tool for image/video
│ └── run_single_image.py # Minimal single-image test script
│
├── models/
│ └── best.pt # Trained YOLOv8 weights (required)
│
├── outputs/
│ ├── annotated.png # Latest annotated image output
│ ├── annotated.mp4 # Latest annotated video output
│ └── state.json # Latest inference summary (counts + timestamp + paths)
│
├── data.yaml # YOLO training config (paths + class names)
├── requirements.txt # Dependencies (streamlit + ultralytics + opencv)
└── README.md


---

## ⚡ Quick Start (Local Setup)

### 1) Install dependencies
```bash
pip install -r requirements.txt


Streamlit Cloud & Linux environments require opencv-python-headless (already included).


2) Run the Web App (Streamlit Dashboard)
streamlit run dashboard/app.py


✅ App features:

Upload image/video from laptop

Adjust confidence threshold

View original + annotated output

Download annotated media

Automatically logs results in outputs/state.json

3) Run CLI Inference (Automation / Testing)
Image inference
python src/run_app.py --input_path "path/to/image.png" --conf 0.5

Video inference
python src/run_app.py --input_path "path/to/video.mp4" --conf 0.5 --save_video


Optional:

python src/run_app.py --input_path "video.mp4" --conf 0.5 --show

🧾 Output Format (Reproducibility)
Annotated outputs

Saved in:

outputs/annotated.png

outputs/annotated.mp4

Structured inference log

outputs/state.json includes:

{
  "timestamp": "YYYY-MM-DD HH:MM:SS",
  "mode": "image/video",
  "input_name": "uploaded_file.mp4",
  "free_count": 12,
  "busy_count": 7,
  "partial_count": 1,
  "output_media_path": "outputs/annotated.mp4"
}


This is useful for:

auditing inference runs

debugging unstable detection

future integration with a backend/database

🧠 Model Training (YOLOv8)
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    name="parking_lot_detector"
)

📊 Model Performance (Validation)
Class	mAP50	mAP50-95
free_parking_space	0.986	0.921
not_free_parking_space	0.994	0.923
partially_free_parking_space	Low	Needs more data

Dataset size: 30 images
Total labeled slots: 903

The partially-free class is significantly harder because it has fewer samples and higher visual similarity.
