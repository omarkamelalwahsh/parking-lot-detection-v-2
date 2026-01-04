# 🚗 Parking Lot Detection & Occupancy Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://parking-lot-detection-v-2-jzd4kygxdn7qtj9eua9lr9.streamlit.app/)

## Executive Summary

**Problem**: Manual monitoring of parking lot occupancy is inefficient and error-prone, leading to wasted time searching for available spots and poor space utilization.

**Solution**: A computer vision system using YOLOv8 for real-time detection of parking slot states (free, occupied, partially occupied) from images and videos, with both CLI and web interfaces.

**Outputs**: Annotated images/videos with color-coded bounding boxes, occupancy counts, and JSON state files for programmatic access.

## Live Demo

Try the interactive web app: [https://parking-lot-detection-v-2-jzd4kygxdn7qtj9eua9lr9.streamlit.app/](https://parking-lot-detection-v-2-jzd4kygxdn7qtj9eua9lr9.streamlit.app/)

Upload images or videos to see real-time detection results.

## Features

### ML Features
- YOLOv8-based object detection for parking slots
- Three-class classification: free_parking_space, not_free_parking_space, partially_free_parking_space
- Confidence thresholding for detection filtering
- Support for images and videos (frame-by-frame processing)

### Engineering Features
- Streamlit web interface for easy uploads and visualization
- CLI tools for batch processing and headless environments
- Portable paths and error handling for cross-platform compatibility
- Automatic file type detection (image vs video)
- Output saving with timestamps and metadata

## System Architecture

```
Input (Image/Video)
    ↓
Load YOLOv8 Model (models/best.pt)
    ↓
Preprocess Frame (resize, normalize)
    ↓
YOLO Inference (predict bounding boxes)
    ↓
Filter by Confidence Threshold
    ↓
Classify & Count Slots (free/busy/partial)
    ↓
Annotate Frame (draw boxes, labels, counts)
    ↓
Display/Save Outputs (annotated media + state.json)
```

## Project Structure

```
parking-lot-detection-v-2-main/
│
├── models/
│   └── best.pt                    # Trained YOLOv8 model weights
│
├── src/
│   ├── run_app.py                 # CLI script for image/video processing
│   ├── run_single_image.py        # Simple CLI for single images
│   └── packages.txt               # Additional packages
│
├── dashboard/
│   └── app.py                     # Streamlit web application
│
├── data/
│   ├── annotations.xml            # Dataset annotations
│   ├── images/                    # Test images
│   ├── labels/                    # YOLO label files (.txt)
│   └── video/                     # Test videos
│
├── outputs/
│   └── state.json                 # Example output state file
│
├── runs/
│   └── parking_lot_detector/      # Training outputs
│       ├── results.csv            # Training results
│       └── weights/
│           ├── best.pt            # Best model checkpoint
│           └── last.pt            # Last model checkpoint
│
├── data.yaml                      # YOLO training configuration
├── requirements.txt               # Python dependencies
├── packages.txt                   # Additional packages
├── runtime.txt                    # Runtime environment
└── README.md                      # This file
```

## Local Setup & Run Instructions

### Prerequisites
- Python 3.10+
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd parking-lot-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Streamlit App

```bash
streamlit run dashboard/app.py
```

- Open the provided URL in your browser.
- Upload an image or video.
- Adjust confidence threshold if needed.
- View annotated results and download outputs.

### Running via CLI

For single images:
```bash
python src/run_single_image.py
# Enter image path when prompted, e.g., C:\path\to\image.png
```

For advanced processing (images/videos with options):
```bash
python src/run_app.py --input_path /path/to/input --conf 0.5 --show --save_video
```

- `--input_path`: Path to image or video file
- `--conf`: Confidence threshold (0.0-1.0)
- `--show`: Display live window (desktop only)
- `--save_video`: Save annotated video output

## Output Formats

### Annotated Media
- **Images**: `outputs/annotated.png` - Original image with overlaid bounding boxes, labels, and occupancy counts.
- **Videos**: `outputs/annotated.mp4` - Annotated video with real-time overlays.

Bounding box colors:
- Green: free_parking_space
- Red: not_free_parking_space
- Yellow: partially_free_parking_space

### State JSON Structure
Saved to `outputs/state.json` after each run:

```json
{
  "timestamp": "2025-12-30 15:59:28",
  "mode": "image",
  "input_path": "/path/to/input",
  "free_count": 4,
  "busy_count": 24,
  "output_media_path": "outputs/annotated.png"
}
```

- `timestamp`: ISO format datetime
- `mode`: "image" or "video"
- `input_path`: Original input file path
- `free_count/busy_count`: Slot counts
- `output_media_path`: Path to annotated file


## Debugging

### Common Errors

- **OpenCV libGL issues (headless environments)**:
  - Error: `libGL.so.1: cannot open shared object file`
  - Solution: Use `opencv-python-headless` instead of `opencv-python` in requirements.txt. Already configured.

- **File upload failures in Streamlit**:
  - Error: "Could not open uploaded video"
  - Solution: Ensure file is valid MP4/AVI/MOV/MKV. Check file size limits in Streamlit Cloud.

- **Threshold instability**:
  - Issue: Counts vary between runs with same input.
  - Solution: Increase confidence threshold (e.g., 0.7) to reduce false positives. For videos, consider frame averaging.

- **Model not found**:
  - Error: "Model file not found: models/best.pt"
  - Solution: Ensure `models/best.pt` exists. Download or retrain if missing.

- **Video processing slow**:
  - Issue: Long processing time.
  - Solution: Reduce video resolution or skip frames (modify code to process every nth frame).

## Roadmap

- **Multi-object tracking**: Integrate DeepSORT or ByteTrack for consistent slot tracking across video frames.
- **ROI masking**: Add polygon-based region of interest to focus detection on specific parking areas.
- **Temporal smoothing**: Implement moving average or Kalman filtering for stable occupancy counts over time.
- **Model improvements**: Expand dataset for better partial slot accuracy and add data augmentation.
- **API deployment**: Expose REST API for integration with other systems.

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and test them.
4. Submit a pull request with a clear description of your changes.

For major changes, please open an issue first to discuss what you would like to change.

## Author

Developed by [omar kamel alwahsh].

For questions or contributions, open an issue or pull request.
