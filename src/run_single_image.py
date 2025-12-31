import cv2
from ultralytics import YOLO
import os

# Path to the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best.pt")
model = YOLO(MODEL_PATH)

def analyze_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("خطأ: لم يتم العثور على الصورة! تأكد من الباث.")
        return

    # Use lower confidence and larger image size for better detection on wide images or small objects
    results = model(img, conf=0.15, imgsz=1024)[0]

    free = busy = partial = 0
    names = model.names

    for box in results.boxes:
        cls = int(box.cls[0])
        label = names[cls]

        if label == "free_parking_space":
            free += 1
            color = (0,255,0)
        elif label == "not_free_parking_space":
            busy += 1
            color = (0,0,255)
        else:
            partial += 1
            color = (0,255,255)

        x1,y1,x2,y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        cv2.putText(img, label, (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    print("============ Parking Summary ============")
    print(f"Free slots    : {free}")
    print(f"Busy slots    : {busy}")
    print(f"Partial slots : {partial}")
    print("==========================================")

    cv2.imwrite("output.png", img)
    print("Output image saved as output.png")

if __name__ == "__main__":
    image_path = input("Enter image path: ").strip().strip('"').strip("'")
    analyze_image(image_path)
