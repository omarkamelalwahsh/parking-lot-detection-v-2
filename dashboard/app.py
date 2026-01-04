import os
import json
import time
from datetime import datetime
import subprocess
import shutil

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


# ✅ state.json بدون Partial
def save_state(mode, input_name, free, busy, output_path):
    state = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "input_name": input_name,
        "free_count": free,
        "busy_count": busy,
        "output_media_path": output_path
    }
    safe_write_json(STATE_FILE, state)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return YOLO(MODEL_PATH)


# ✅ بدون Partial
def annotate_frame(frame_bgr, results, names, conf_threshold):
    free, busy = 0, 0

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue

        cls = int(box.cls[0])
        label = names.get(cls, str(cls))

        if label == "free_parking_space":
            free += 1
            color = (0, 255, 0)  # green
        elif label == "not_free_parking_space":
            busy += 1
            color = (0, 0, 255)  # red
        else:
            continue  # ignore any other class

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame_bgr,
            f"{label} {conf:.2f}",
            (x1, max(15, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    info_text = f"Free: {free} | Busy: {busy}"
    cv2.putText(
        frame_bgr,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return frame_bgr, free, busy


def read_uploaded_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr


def save_image(path, img_bgr):
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img_bgr)


def get_file_type(filename: str):
    ext = os.path.splitext(filename)[1].lower()
    if ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        return "image"
    if ext in [".mp4", ".avi", ".mov", ".mkv"]:
        return "video"
    return None


# ✅ Auto Confidence helper
def compute_auto_conf(results_list, margin=0.10, default=0.50):
    confs = []
    for res in results_list:
        if res is None or res.boxes is None:
            continue
        for box in res.boxes:
            confs.append(float(box.conf[0]))

    if len(confs) == 0:
        return default

    med = float(np.median(confs))
    auto_conf = max(0.05, min(0.95, med - margin))
    return auto_conf


# ✅ Convert MP4 to H264 (browser playable)
def convert_to_h264(input_path, output_path):
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        # ffmpeg not found, return raw output
        st.warning("FFmpeg not found. Video will be saved as raw MP4 (may not play in browser).")
        return input_path

    cmd = [
        ffmpeg_path,
        "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "23",
        output_path
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path


# =========================
# ✅ Streamlit UI
# =========================
st.set_page_config(page_title="Parking Lot Detection", layout="wide")
st.title("🚗 Parking Lot Detection (YOLOv8)")

ensure_dir(OUTPUT_DIR)

model = load_model()
if model is None:
    st.error("Model file not found: models/best.pt")
    st.stop()

names = model.names


# =========================
# ✅ Sidebar Settings
# =========================
st.sidebar.header("Settings")

st.sidebar.subheader("Confidence Threshold")

conf_mode = st.sidebar.radio(
    "Choose mode:",
    ["Manual", "Auto (recommended)"],
    index=1
)

# ✅ Manual slider يظهر فقط في manual
manual_conf = None
if conf_mode == "Manual":
    manual_conf = st.sidebar.slider(
        "Manual confidence threshold",
        0.05, 0.95, 0.50, 0.05
    )

# ✅ Auto settings ثابتة (مش ظاهرة للمستخدم)
AUTO_MARGIN = 0.10
AUTO_SAMPLE_FRAMES = 30
AUTO_DEFAULT = 0.50

save_outputs = st.sidebar.checkbox("Save outputs to outputs/", value=True)

st.sidebar.markdown("---")
st.sidebar.info("Upload an image or a video. The app will detect the file type automatically.")


# =========================
# ✅ ONE Upload (Image or Video)
# =========================
uploaded_file = st.file_uploader(
    "Upload an Image or Video",
    type=["png", "jpg", "jpeg", "bmp", "mp4", "avi", "mov", "mkv"]
)

if uploaded_file is not None:
    input_name = uploaded_file.name
    file_type = get_file_type(input_name)

    if file_type is None:
        st.error("Unsupported file type.")
        st.stop()

    # =========================
    # ✅ IMAGE FLOW
    # =========================
    if file_type == "image":
        img_bgr = read_uploaded_image(uploaded_file)

        if conf_mode == "Manual":
            conf_threshold = manual_conf
        else:
            temp_res = model.predict(img_bgr, conf=0.05, verbose=False)[0]
            conf_threshold = compute_auto_conf(
                [temp_res],
                margin=AUTO_MARGIN,
                default=AUTO_DEFAULT
            )

        st.info(f"Using confidence threshold = {conf_threshold:.2f}")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

        results = model.predict(img_bgr, conf=conf_threshold, verbose=False)[0]
        annotated_bgr, free, busy = annotate_frame(img_bgr.copy(), results, names, conf_threshold)

        with col2:
            st.subheader("Annotated")
            st.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

        st.success(f"Done ✅  Free={free}, Busy={busy}")

        out_path = os.path.join(OUTPUT_DIR, "annotated.png")
        if save_outputs:
            save_image(out_path, annotated_bgr)
            save_state("image", input_name, free, busy, out_path)

        st.download_button(
            "Download annotated image",
            data=cv2.imencode(".png", annotated_bgr)[1].tobytes(),
            file_name="annotated.png",
            mime="image/png"
        )

    # =========================
    # ✅ VIDEO FLOW
    # =========================
    else:
        temp_video_path = os.path.join(OUTPUT_DIR, f"temp_{int(time.time())}_{input_name}")
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.subheader("Original Video")
        st.video(temp_video_path)

        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            st.error("Could not open uploaded video.")
            st.stop()

        # ✅ Determine confidence threshold
        if conf_mode == "Manual":
            conf_threshold = manual_conf
        else:
            st.info("Auto mode: computing confidence threshold from first frames...")

            sample_results = []
            sample_count = 0

            while sample_count < AUTO_SAMPLE_FRAMES:
                ret, frame = cap.read()
                if not ret:
                    break

                res = model.predict(frame, conf=0.05, verbose=False)[0]
                sample_results.append(res)
                sample_count += 1

            conf_threshold = compute_auto_conf(
                sample_results,
                margin=AUTO_MARGIN,
                default=AUTO_DEFAULT
            )

            st.info(f"Auto confidence threshold computed = {conf_threshold:.2f}")

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 25

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        st.warning("Processing full video... this may take some time.")

        raw_out_path = os.path.join(OUTPUT_DIR, "annotated_raw.mp4")
        final_out_path = os.path.join(OUTPUT_DIR, "annotated.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(raw_out_path, fourcc, fps, (width, height))

        if not out.isOpened():
            st.error("VideoWriter failed to open. Codec not supported.")
            cap.release()
            st.stop()

        progress = st.progress(0)
        start_time = time.time()

        last_counts = (0, 0)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
            annotated, free, busy = annotate_frame(frame.copy(), results, names, conf_threshold)
            last_counts = (free, busy)

            out.write(annotated)

            if total_frames > 0 and frame_count % 5 == 0:
                progress.progress(min(frame_count / total_frames, 1.0))

        cap.release()
        out.release()
        progress.progress(1.0)

        # ✅ Convert to H264 for Streamlit playback
        st.info("Converting video to web-compatible format (H.264)...")
        playable_path = convert_to_h264(raw_out_path, final_out_path)

        free, busy = last_counts
        st.success(f"Done ✅  Free={free}, Busy={busy}")
        st.info(f"Processed {frame_count} frames in {time.time() - start_time:.1f} seconds.")

        if save_outputs:
            save_state("video", input_name, free, busy, playable_path)

        st.subheader("Annotated Video")
        st.video(playable_path)

        with open(playable_path, "rb") as f:
            st.download_button(
                "Download annotated video",
                data=f.read(),
                file_name="annotated.mp4",
                mime="video/mp4"
            )

else:
    st.info("⬆️ Upload an image or a video to start detection.")
