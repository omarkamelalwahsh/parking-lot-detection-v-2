import os
import json
import time
from datetime import datetime

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO


# =========================
# ✅ Paths (portable)
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
STATE_FILE = os.path.join(OUTPUT_DIR, "state.json")


# =========================
# ✅ Helpers
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_write_json(path, data):
    try:
        ensure_dir(os.path.dirname(path))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Failed to write JSON: {e}")
        return False


def save_state(mode, input_name, free, busy, partial, output_path):
    state = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "input_name": input_name,
        "free_count": free,
        "busy_count": busy,
        "partial_count": partial,
        "output_media_path": output_path
    }
    safe_write_json(STATE_FILE, state)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return YOLO(MODEL_PATH)


def annotate_frame(frame_bgr, results, names, conf_threshold):
    """Draw YOLO boxes + count classes based on label names."""
    free, busy, partial = 0, 0, 0

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue

        cls = int(box.cls[0])
        label = names.get(cls, str(cls))

        # Default: yellow
        color = (0, 255, 255)
        if label == "free_parking_space":
            free += 1
            color = (0, 255, 0)
        elif label == "not_free_parking_space":
            busy += 1
            color = (0, 0, 255)
        else:
            partial += 1

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame_bgr, f"{label} {conf:.2f}", (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    info_text = f"Free: {free} | Busy: {busy} | Partial: {partial}"
    cv2.putText(frame_bgr, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return frame_bgr, free, busy, partial


def read_uploaded_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr


def save_image(path, img_bgr):
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img_bgr)


def save_video_from_frames(path, frames_bgr, fps=25):
    ensure_dir(os.path.dirname(path))
    if not frames_bgr:
        return False
    h, w = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames_bgr:
        out.write(f)
    out.release()
    return True


# =========================
# ✅ Streamlit UI
# =========================
st.set_page_config(page_title="Parking Lot Detection", layout="wide")
st.title("🚗 Parking Lot Detection (YOLOv8)")

ensure_dir(OUTPUT_DIR)

model = load_model()
if model is None:
    st.error("Model file not found: models/best.pt")
    st.info("Please upload best.pt to models/ folder OR provide a download link and load it automatically.")
    st.stop()

names = model.names

# Sidebar settings
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.5, 0.05)
save_outputs = st.sidebar.checkbox("Save outputs to outputs/", value=True)

mode = st.sidebar.radio("Choose input type", ["Image", "Video"])

# =========================
# ✅ Image Mode
# =========================
if mode == "Image":
    uploaded_img = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])
    if uploaded_img is not None:
        input_name = uploaded_img.name
        img_bgr = read_uploaded_image(uploaded_img)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

        # YOLO inference
        results = model.predict(img_bgr, conf=conf_threshold, verbose=False)[0]
        annotated_bgr, free, busy, partial = annotate_frame(img_bgr.copy(), results, names, conf_threshold)

        with col2:
            st.subheader("Annotated")
            st.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

        st.success(f"Done ✅  Free={free}, Busy={busy}, Partial={partial}")

        if save_outputs:
            out_path = os.path.join(OUTPUT_DIR, "annotated.png")
            save_image(out_path, annotated_bgr)
            save_state("image", input_name, free, busy, partial, out_path)
            st.info(f"Saved annotated image to: {out_path}")
            st.download_button("Download annotated image",
                               data=cv2.imencode(".png", annotated_bgr)[1].tobytes(),
                               file_name="annotated.png",
                               mime="image/png")

# =========================
# ✅ Video Mode
# =========================
else:
    uploaded_vid = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_vid is not None:
        input_name = uploaded_vid.name

        # Save uploaded video temporarily
        temp_video_path = os.path.join(OUTPUT_DIR, "temp_upload_video.mp4")
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_vid.read())

        st.video(temp_video_path)

        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            st.error("Could not open uploaded video.")
            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 25

        st.warning("Processing video... this may take some time depending on video length.")

        frames_out = []
        last_counts = (0, 0, 0)

        frame_limit = st.sidebar.number_input("Max frames to process (0 = all)", min_value=0, max_value=5000, value=300)

        frame_count = 0
        progress = st.progress(0)
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_limit and frame_count > frame_limit:
                break

            results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
            annotated, free, busy, partial = annotate_frame(frame.copy(), results, names, conf_threshold)
            last_counts = (free, busy, partial)
            frames_out.append(annotated)

            if frame_count % 10 == 0:
                progress.progress(min(frame_count / max(1, frame_limit if frame_limit else frame_count), 1.0))

        cap.release()
        progress.progress(1.0)

        free, busy, partial = last_counts
        st.success(f"Done ✅ Free={free}, Busy={busy}, Partial={partial}")
        st.info(f"Processed {frame_count} frames in {time.time() - start_time:.1f} seconds.")

        if save_outputs:
            out_video_path = os.path.join(OUTPUT_DIR, "annotated.mp4")
            ok = save_video_from_frames(out_video_path, frames_out, fps=fps)
            if ok:
                save_state("video", input_name, free, busy, partial, out_video_path)
                st.info(f"Saved annotated video to: {out_video_path}")

                with open(out_video_path, "rb") as f:
                    st.download_button(
                        "Download annotated video",
                        data=f.read(),
                        file_name="annotated.mp4",
                        mime="video/mp4"
                    )
            else:
                st.error("Failed to save output video.")
