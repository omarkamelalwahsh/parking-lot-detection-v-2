import cv2
import argparse
import os
import json
import time
from datetime import datetime
from ultralytics import YOLO


# =========================
# ✅ Paths (Windows-safe)
# =========================
# project root = parent of src/
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
STATE_FILE = os.path.join(OUTPUT_DIR, "state.json")


# =========================
# ✅ Helpers
# =========================
def ensure_output_dir():
    """Always create outputs folder inside project root."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_path(p: str) -> str:
    """Remove extra quotes/spaces from Windows paths."""
    return p.strip().strip('"').strip("'").strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Parking Lot Detection Inference Script")
    parser.add_argument("--input_path", type=str, help="Path to input image or video")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--show", action="store_true", help="Show live window for video")
    parser.add_argument("--save_video", action="store_true", help="Save annotated video output")

    args = parser.parse_args()

    # ✅ Ask user if input_path missing
    if not args.input_path:
        args.input_path = input("Enter input path (image/video): ")

    args.input_path = clean_path(args.input_path)
    return args


def get_input_mode(path):
    ext = os.path.splitext(path)[1].lower()
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    vid_exts = {'.mp4', '.avi', '.mov', '.mkv'}
    if ext in img_exts:
        return "image"
    if ext in vid_exts:
        return "video"
    return None


def safe_write_json(path, data):
    """Write JSON safely to outputs/state.json"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to write JSON to {path}")
        print("[ERROR] Exception:", e)
        return False


def safe_imwrite(path, img):
    """Save image safely inside outputs folder."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ok = cv2.imwrite(path, img)
        if not ok:
            raise RuntimeError("cv2.imwrite returned False")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save image to {path}")
        print("[ERROR] Exception:", e)
        return False


def save_state(mode, input_path, free, busy, partial, output_path):
    state = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "input_path": input_path,
        "free_count": free,
        "busy_count": busy,
        "partial_count": partial,
        "output_media_path": output_path
    }
    ok = safe_write_json(STATE_FILE, state)
    if ok:
        print(f"[INFO] state.json saved -> {STATE_FILE}")


def annotate_frame(frame, results, model, conf_threshold):
    free, busy, partial = 0, 0, 0
    names = model.names

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue

        cls = int(box.cls[0])
        label = names.get(cls, str(cls))

        # Colors
        color = (0, 255, 255)  # yellow
        if label == "free_parking_space":
            free += 1
            color = (0, 255, 0)  # green
        elif label == "not_free_parking_space":
            busy += 1
            color = (0, 0, 255)  # red
        else:
            partial += 1

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # overlay counts
    info_text = f"Free: {free} | Busy: {busy} | Partial: {partial} | {datetime.now().strftime('%H:%M:%S')}"
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2, cv2.LINE_AA)

    return frame, free, busy, partial


# =========================
# ✅ Image Mode
# =========================
def process_image(input_path, model, conf_threshold):
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"[ERROR] Could not read image: {input_path}")
        return

    results = model(frame, conf=conf_threshold)[0]
    annotated, free, busy, partial = annotate_frame(frame, results, model, conf_threshold)

    output_path = os.path.join(OUTPUT_DIR, "annotated.png")
    if safe_imwrite(output_path, annotated):
        save_state("image", input_path, free, busy, partial, output_path)

    print(f"[DONE] Image processed -> Free={free}, Busy={busy}, Partial={partial}")
    print(f"[DONE] Annotated saved -> {output_path}")


# =========================
# ✅ Video Mode
# =========================
def process_video(input_path, model, conf_threshold, show, save_video):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    output_path = os.path.join(OUTPUT_DIR, "annotated.mp4")
    out = None

    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    last_state_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf_threshold)[0]
        annotated, free, busy, partial = annotate_frame(frame, results, model, conf_threshold)

        if save_video and out is not None:
            out.write(annotated)

        # save JSON state every 1 second
        now = time.time()
        if now - last_state_time >= 1.0:
            save_state("video", input_path, free, busy, partial, output_path if save_video else "NOT_SAVED")
            last_state_time = now

        if show:
            cv2.imshow("Parking Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if out is not None:
        out.release()
    if show:
        cv2.destroyAllWindows()

    save_state("video", input_path, free, busy, partial, output_path if save_video else "NOT_SAVED")
    print("[DONE] Video processed.")
    if save_video:
        print(f"[DONE] Annotated video saved -> {output_path}")
    else:
        print("[DONE] Video output not saved (use --save_video).")


# =========================
# ✅ Main
# =========================
def main():
    ensure_output_dir()
    args = parse_args()

    if not os.path.exists(args.input_path):
        print(f"[ERROR] Input does not exist: {args.input_path}")
        return

    mode = get_input_mode(args.input_path)
    if not mode:
        print(f"[ERROR] Unsupported file type: {args.input_path}")
        return

    print("BASE_DIR:", BASE_DIR)
    print("OUTPUT_DIR:", OUTPUT_DIR)
    print("STATE_FILE:", STATE_FILE)
    print(f"Loading model from: {MODEL_PATH}")

    model = YOLO(MODEL_PATH)

    if mode == "image":
        process_image(args.input_path, model, args.conf)
    else:
        process_video(args.input_path, model, args.conf, args.show, args.save_video)


if __name__ == "__main__":
    main()
