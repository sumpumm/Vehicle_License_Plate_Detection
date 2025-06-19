from ultralytics import YOLO
import cv2
from utils import get_xyxy

video_path=r"test_vids\test2.mp4"

#load image
# image = cv2.imread(r"test_vids\3vehicles.jpg")

#load video
vid=cv2.VideoCapture(video_path)

# Load trained YOLOv8 model
model = YOLO(r"runs/detect/train/weights/best.pt")


# Run tracking on video with ByteTrack
# results = model.track(
#     source=r"test_vids/test1.mp4",  
#     save=True,                       
#     project="outputs",               
#     name="yolo_result",            
#     tracker="bytetrack.yaml",                   
#     save_crop=True                
# )

#,save=True,project="outputs",name="yolo_result",save_crop=True
results=model.track(source=video_path,stream=True)


#get the id and coordinates of the detected numberplates
plates_dict=get_xyxy(results)

for id,coord in plates_dict.items():
    x1,y1,x2,y2,frame_number=coord
    
    vid.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret,frame=vid.read()
    if not ret:
        print(f"Error: Could not read frame {frame_number}")
    else:
        crop = frame[y1:y2, x1:x2]
        cv2.imwrite(f"outputs/extracted_plate/plate_{id}.jpg", crop)


