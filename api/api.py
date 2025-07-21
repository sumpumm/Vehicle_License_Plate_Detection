from fastapi import FastAPI
from models import ModelInput
from ultralytics import YOLO
import cv2,easyocr
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import get_xyxy

app=FastAPI()

@app.post("/api/detect")
def detect(inputs:ModelInput):
    vid=cv2.VideoCapture(inputs.filePath)
    model = YOLO(r"../runs/detect/train/weights/best.pt")
    reader=easyocr.Reader(['ne'])
    results=model.track(source=inputs.filePath,stream=True)

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
            ocr_result=reader.readtext(crop)
            for res in ocr_result:
                text = res[1]
                detected_plates_ocr_result.append(text)
            cv2.imwrite(f"outputs/extracted_plate/plate_{id}.jpg", crop)
    return {"result":detected_plates_ocr_result}       
    