from ultralytics import YOLO


model = YOLO("yolov8n.yaml")

results = model.train(data="F:\SAMARPAN\Vehicle Number Plate Detection\config.yaml", epochs=40)