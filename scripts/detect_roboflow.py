import os

import cv2
from dotenv import load_dotenv
from inference import get_model

load_dotenv()

MODEL_ID = "detectv2-jf4nn/1"
CONFIDENCE = 0.5
CAMERA_INDEX = 0  # change to 1 if the wrong camera opens

# Class names this model was trained on, compared in lowercase.
RECYCLE = {"bottle", "paper", "box", "can", "recyclable"}

COLOR_RECYCLE = (0, 200, 0)  # green (OpenCV uses BGR order)
COLOR_TRASH = (0, 0, 255)    # red


def classify_bin(class_name):
    """Return the bin label and box color for a detected class."""
    if class_name.strip().lower() in RECYCLE:
        return "Recycle", COLOR_RECYCLE
    return "Trash", COLOR_TRASH


def draw_labeled_box(img, x1, y1, x2, y2, label, color):
    """Draw a box with a filled label tag above it."""
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def main():
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Missing ROBOFLOW_API_KEY. Put it in a .env file next to this script.")

    # First run downloads the weights and caches them; after that it runs locally.
    model = get_model(model_id=MODEL_ID, api_key=api_key)
    print("[INFO] Model loaded:", MODEL_ID)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Could not read from the camera.")
            break
        frame = cv2.flip(frame, 1)

        result = model.infer(frame, confidence=CONFIDENCE)[0]

        for pred in result.predictions:
            # Roboflow gives the box center plus width and height.
            x1 = int(pred.x - pred.width / 2)
            y1 = int(pred.y - pred.height / 2)
            x2 = int(pred.x + pred.width / 2)
            y2 = int(pred.y + pred.height / 2)

            tag, color = classify_bin(pred.class_name)
            label = f"{tag}: {pred.class_name} {pred.confidence:.2f}"
            draw_labeled_box(frame, x1, y1, x2, y2, label, color)

        cv2.imshow("Trash Sorter", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()