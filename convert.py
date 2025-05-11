from ultralytics import YOLO

# Load YOLO model
model = YOLO("D:/Minor_Project-2/pythonProject/runs/detect/train/weights/best.pt")

# Export to ONNX format
model.export(format="onnx", dynamic=True)  # Dynamic batch size
