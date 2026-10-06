import cv2
from pathlib import Path
from datetime import datetime

# Dataset root
DATASET_DIR = Path(__file__).resolve().parent.parent

# Change this to aroshi, hiruni, or unknown
CLASS_NAME = "aroshi"

# Save images inside raw/class_name
SAVE_DIR = DATASET_DIR / "raw" / CLASS_NAME
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Haar Cascade
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

print(f"Capturing images for: {CLASS_NAME}")
print("Press SPACE to capture an image.")
print("Press Q to quit.")

count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Cannot read webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        # 20% padding
        padding_x = int(w * 0.20)
        padding_y = int(h * 0.20)

        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(frame.shape[1], x + w + padding_x)
        y2 = min(frame.shape[0], y + h + padding_y)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    # SPACE = capture
    if key == 32:

        if len(faces) > 0:

            x, y, w, h = faces[0]

            padding_x = int(w * 0.20)
            padding_y = int(h * 0.20)

            x1 = max(0, x - padding_x)
            y1 = max(0, y - padding_y)
            x2 = min(frame.shape[1], x + w + padding_x)
            y2 = min(frame.shape[0], y + h + padding_y)

            face = frame[y1:y2, x1:x2]

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = SAVE_DIR / f"{CLASS_NAME}_{timestamp}.jpg"

            cv2.imwrite(str(filename), face)

            count += 1

            print(f"Saved: {filename}")

        else:
            print("No face detected.")

    # Q = quit
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nFinished. Captured {count} new images.")