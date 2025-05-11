from ultralytics import YOLO

model = YOLO('yolov8n.pt')

def main():
    model.train(
        data="D:/Minor_Project-2/pythonProject/Dataset/SplitData/data.yaml",
        epochs=35,
        batch=4,  # Reduce batch size to avoid OOM (Out of Memory) errors
        imgsz=416,  # Reduce image size to lower GPU memory usage
        amp=True,  # Enable Automatic Mixed Precision (AMP) to optimize memory
        workers=2  # Reduce number of workers to prevent excessive RAM usage
    )


if __name__ == '__main__':
    main()