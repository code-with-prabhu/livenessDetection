import math
import time
import cv2
import cvzone
from ultralytics import YOLO

confidence_threshold = 0.65 

cap = cv2.VideoCapture(0)  # Webcam Capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)  # Reduce size for better FPS
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

model = YOLO("D:/Minor_Project-2/pythonProject/runs/detect/train/weights/best.pt")

classNames = ["fake", "real"]

prev_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break  # Stop if no frame is captured

    results = model(img, stream=True, verbose=False)

    for r in results:
        for box in r.boxes:
            # Bounding Box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            # Confidence & Class
            conf = round(float(box.conf[0]), 2)  # Convert to float and round
            cls = int(box.cls[0])

            # Apply confidence threshold logic
            if conf < confidence_threshold:
                detected_class = "fake"
                color = (0, 0, 255)  # Red for fake
            else:
                detected_class = classNames[cls]  # Use original class
                color = (0, 255, 0) if detected_class == "real" else (0, 0, 255)

            # Draw bounding box and label
            cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
            cvzone.putTextRect(img, f'{detected_class.upper()} {int(conf*100)}%',
                               (max(0, x1), max(35, y1)), scale=2, thickness=4, colorR=color, colorB=color)

    # FPS Calculation
    fps = 1 / max(0.0001, (new_frame_time - prev_frame_time))
    prev_frame_time = new_frame_time
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Image", img)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
