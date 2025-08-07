from ultralytics import YOLO

model = YOLO(r"../Character_Detector/detect/train/weights/best.pt")

result=model.predict(source="viber_image_2025-08-02_22-07-34-144.jpg",save=True)