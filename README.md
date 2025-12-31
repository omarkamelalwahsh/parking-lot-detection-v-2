# 🚗 Parking Lot Detection & Occupancy Demo

This project is a **Machine Learning / Computer Vision system** that analyzes parking lot images and detects:

- 🟩 Free parking slots  
- 🟥 Occupied parking slots  
- 🟨 Partially free slots

The model is trained using **YOLOv8** on a custom parking lot dataset.

---

## 🚀 Features

- 📊 Interactive Streamlit Dashboard  
- Detects free / busy / partial parking slots  
- Works on both images and videos  
- Fast inference (CPU supported)  
- Real-time visualization with color-coded bounding boxes  

---

## 📁 Project Structure

parking-lot-detection/
│
├── models/
│ └── best.pt → YOLO model weights
│
├── src/
│ ├── run_single_image.py → Main script to test images
│ ├── slot_prediction.py → Predicts car movement (leaving, stationary)
│ └── utils.py → Helper functions
│
├── data/
│ ├── images/ → Put your test images here
│ └── labels/ → YOLO annotation labels
│
├── runs/ → YOLO training output folder
│
├── README.md
└── requirements.txt

yaml

---

## 🖼️ Example Results

Put your result images in:

data/images/

yaml

Then reference them below:

### Example 1  

<img src="data/images/1.png" width="500">

### Example 2  

<img src="data/images/2.png" width="500">

---

## ▶️ How to Run the Project

### 1) Install dependencies

pip install ultralytics opencv-python numpy

csharp

Or install using requirements file:

pip install -r requirements.txt

yaml

---

### 2) Run inference on an image

python run_single_image.py

yaml

The script will ask you:

Enter image path:

makefile

Example:

C:\Users\User\Parking Lot Dataset\data\images\2.png

yaml

The program will:

- Count free / busy / partial slots  
- Display the image with bounding boxes  

### 3) Run the Interactive Dashboard

To run the dashboard (supports images and videos):

```powershell
python -m streamlit run dashboard/app.py
```

*Note: If you are in WSL, use `python3 -m streamlit run dashboard/app.py`.*

---

## 🧠 YOLOv8 Training Code

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    name="parking_lot_detector"
)
```

📊 **Model Performance**

| Class | mAP50 | mAP50-95 |
|-------|-------|----------|
| free_parking_space | 0.986 | 0.921 |
| not_free_parking_space | 0.994 | 0.923 |
| partially_free_parking_space | Low | (needs more data) |

- **Dataset size**: 30 images
- **Total labeled slots**: 903

🔮 **Future Improvements**

- Multi-object tracking (DeepSORT, BYTETrack)
- Improved annotations for partial slots
- Slot polygon detection
