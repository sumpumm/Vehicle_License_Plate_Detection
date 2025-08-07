from fastapi import FastAPI
from models import ModelInput
import sys,os
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from classes.Video import Video
from classes.Frame import Frame

from utils import get_xyxy,convert_yolo_output_avi_to_mp4
app=FastAPI()
origins = [ "*", ]
app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,   
        allow_methods=["*"],       
        allow_headers=["*"],    
    )
app.mount("/tracks", StaticFiles(directory="outputs/track"), name="tracks")


# @app.post("/api/detect")
# def detect(inputs:ModelInput):
#     vid=cv2.VideoCapture(inputs.filePath)
#     model = YOLO(r"../runs/detect/train/weights/best.pt")
#     reader=easyocr.Reader(['ne'])
#     results=model.track(source=inputs.filePath,stream=True,save=True,conf=0.6,project="outputs",name="track",exist_ok=True )
    
#     #get the id and coordinates of the detected numberplates
#     plates_dict=get_xyxy(results)

#     detected_plates_ocr_result=[]

#     for id,coord in plates_dict.items():
#         x1,y1,x2,y2,frame_number=coord
    
#         vid.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
#         ret,frame=vid.read()
#         if not ret:
#             print(f"Error: Could not read frame {frame_number}")
#         else:
#             crop = frame[y1:y2, x1:x2]
#             ocr_result=reader.readtext(crop)
#             for res in ocr_result:
#                 text = res[1]
#                 detected_plates_ocr_result.append(text)
#             cv2.imwrite(f"outputs/extracted_plate/plate_{id}.jpg", crop)
            
            
#     convert_yolo_output_avi_to_mp4("outputs","track",inputs.fileName.replace(".mp4", ".avi"))
#     return {"result":detected_plates_ocr_result}       
    
    
@app.post("/api/detect")
def detect(inputs:ModelInput):
    video = Video(inputs.filePath, "../runs/detect/train/weights/best.pt")
    results = video.process_video()
    plates_dict = video.get_plate_coordinates()
    
    processor = Frame()
    detected_plates_ocr_result = []
    
    for id, coord in plates_dict.items():
        x1, y1, x2, y2, frame_number = coord

        frame = video.get_frame(frame_number)
        if not frame is None:
            texts = processor.extract_plate(frame, (x1, y1, x2, y2), id)
            for text in texts:
                detected_plates_ocr_result.append(text)
        else:
            print(f"Error: Could not read frame {frame_number}")
    convert_yolo_output_avi_to_mp4("outputs","track",inputs.fileName.replace(".mp4", ".avi"))
    return {"result":detected_plates_ocr_result}


