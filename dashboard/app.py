import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

# Page configuration
st.set_page_config(
    page_title="Parking Lot Detector",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS for glassmorphism and premium look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path):
    """Load YOLO model and cache it."""
    if not os.path.exists(model_path):
        return None
    return YOLO(model_path)

def process_frame(frame, model, conf_threshold):
    """Run inference on a single frame and return annotated frame and counts."""
    results = model(frame, conf=conf_threshold, verbose=False)[0]
    
    counts = {
        "free_parking_space": 0,
        "not_free_parking_space": 0,
        "partially_free_parking_space": 0
    }
    
    annotated_frame = frame.copy()
    
    # Define colors (BGR)
    colors = {
        "free_parking_space": (0, 255, 0),       # Green
        "not_free_parking_space": (0, 0, 255),   # Red
        "partially_free_parking_space": (0, 255, 255) # Yellow
    }
    
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = results.names[cls_id]
        conf = float(box.conf[0])
        
        if label in counts:
            counts[label] += 1
            
            # Draw box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = colors.get(label, (255, 255, 255))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            text = f"{label} {conf:.2f}"
            cv2.putText(annotated_frame, text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    return annotated_frame, counts

def main():
    st.title("🚗 Parking Lot Occupancy Dashboard")
    st.markdown("---")

    # Sidebar for controls
    st.sidebar.header("Controls")
    input_path = st.sidebar.text_input("Input Path (Image or Video)", placeholder="e.g. data/sample.jpg")
    if input_path:
        # Strip quotes that might be added when copying paths in Windows
        input_path = input_path.strip('"').strip("'")
        
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    
    is_video = False
    if input_path:
        ext = os.path.splitext(input_path)[1].lower()
        if ext in ['.mp4', '.avi', '.mov', '.mkv']:
            is_video = True
            frame_skip = st.sidebar.number_input("Process 1 frame every N frames", min_value=1, value=5)
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            is_video = False
        else:
            st.error("Unsupported file extension. Please use an image or video path.")
            return

    run_button = st.sidebar.button("Run Detection")

    # Metrics container
    metric_cols = st.columns(3)
    free_metric = metric_cols[0].empty()
    busy_metric = metric_cols[1].empty()
    partial_metric = metric_cols[2].empty()

    # Image/Video display container
    display_container = st.empty()

    if run_button:
        if not input_path:
            st.error("Please provide an input path.")
            return
        
        if not os.path.exists(input_path):
            st.error(f"Path does not exist: {input_path}")
            return

        model = load_model("models/best.pt")
        if model is None:
            st.error("Model file 'models/best.pt' not found.")
            return

        if not is_video:
            # Process Image
            frame = cv2.imread(input_path)
            if frame is None:
                st.error("Failed to load image.")
                return
            
            annotated_frame, counts = process_frame(frame, model, conf_threshold)
            
            # Update metrics
            free_metric.metric("Free", counts["free_parking_space"])
            busy_metric.metric("Busy", counts["not_free_parking_space"], delta_color="inverse")
            partial_metric.metric("Partial", counts["partially_free_parking_space"])
            
            # Display image
            display_container.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            
        else:
            # Process Video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                st.error("Failed to open video.")
                return
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_skip == 0:
                    annotated_frame, counts = process_frame(frame, model, conf_threshold)
                    
                    # Update metrics
                    free_metric.metric("Free", counts["free_parking_space"])
                    busy_metric.metric("Busy", counts["not_free_parking_space"], delta_color="inverse")
                    partial_metric.metric("Partial", counts["partially_free_parking_space"])
                    
                    # Display frame
                    display_container.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                frame_count += 1
                
            cap.release()
            st.success("Video processing complete.")

if __name__ == "__main__":
    main()
