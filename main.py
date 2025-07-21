from ultralytics import YOLO
import cv2,easyocr
from utils import get_xyxy
import pandas as pd

video_path=r"test_vids\test2.mp4"

vid=cv2.VideoCapture(video_path)

# Load trained YOLOv8 model
model = YOLO(r"runs/detect/train/weights/best.pt")
reader=easyocr.Reader(['ne'])


#,save=True,project="outputs",name="yolo_result",save_crop=True
results=model.track(source=video_path,stream=True)

#get the id and coordinates of the detected numberplates
plates_dict=get_xyxy(results)

detected_plates_ocr_result=[]

for id,coord in plates_dict.items():
    x1,y1,x2,y2,frame_number=coord
    
    vid.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret,frame=vid.read()
    if not ret:
        print(f"Error: Could not read frame {frame_number}")
    else:
        crop = frame[y1:y2, x1:x2]
        # license_plate_crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
        ocr_result=reader.readtext(crop)
        for res in ocr_result:
            text = res[1]
            detected_plates_ocr_result.append(text)
        cv2.imwrite(f"outputs/extracted_plate/plate_{id}.jpg", crop)
        
print(detected_plates_ocr_result)               
df=pd.DataFrame({
    "plate_number":detected_plates_ocr_result
})
df.to_csv('plates.csv', index=False)

