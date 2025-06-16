from ultralytics import YOLO


model = YOLO(r"runs/detect/train/weights/best.pt")  


results = model.predict(
    source=r"test_vids/test1.mp4",  
    save=True,
    project="outputs",              
    name="yolo_result")


