from ultralytics import YOLO
model = YOLO('yolov8n-cls.pt')

model.train(
    data='dataset',    # Path to your dataset folder
    epochs=10,         # Number of training epochs (you can change later)
    imgsz=224,         # Image size
    batch=32           # Batch size
)