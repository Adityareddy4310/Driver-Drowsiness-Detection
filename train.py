from ultralytics import YOLO

# Start with small & fast model (good for beginners)
model = YOLO("yolov8n.pt")   

# Train
model.train(
    data="dataset/data.yaml",     # your yaml file
    epochs=40,                    # 30–60 is usually enough
    imgsz=640,
    batch=8,                     # reduce to 8 if computer is slow / low RAM
    name="drowsy_yolo",
    patience=20                   # stop early if no improvement
)