from fastapi import FastAPI, HTTPException
import base64,io
from io import BytesIO
from PIL import Image
import numpy as np
from models import ImageInput
from ultralytics import YOLO
from fastapi.middleware.cors import CORSMiddleware
import easyocr,cv2

model = YOLO(r"../runs/detect/train/weights/best.pt")
reader = easyocr.Reader(['ne'])

app=FastAPI()
origins = [ "*", ]
app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,   
        allow_methods=["*"],       
        allow_headers=["*"],    
    )

@app.post("/api/webcam")
async def upload_frame(data: ImageInput):
    try:
        # Strip header if present (e.g., data:image/jpeg;base64,...)
        header, encoded = data.image.split(",", 1) if "," in data.image else ("", data.image)

        # Decode the base64 string
        image_bytes = base64.b64decode(encoded)

        # Convert to PIL Image
        image = Image.open(BytesIO(image_bytes))

        # Optional: convert to OpenCV format (for your license plate detection)
        image_np = np.array(image)
        

        #  Now you can run your license plate detection on image_cv
        results=model.track(source=image_np,conf=0.6,project="outputs",name="track",exist_ok=True)
        boxes=results[0].boxes.xyxy.numpy()
        x1, y1, x2, y2 = map(int, boxes[0])
        
        annotated_img = results[0].plot()
        
        annotated_pil = Image.fromarray(annotated_img)
        
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        cropped_plate=image_bgr[y1:y2, x1:x2]
        
        cv2.imwrite("outputs/track/cropped_plate.jpg", cropped_plate)
        
        ocr_results = reader.readtext(cropped_plate)
        texts = [res[1] for res in ocr_results]
        conc_text = ""
    
        for text in texts:
            conc_text+=text
        
        with open("outputs/track/plate_text.txt", "a", encoding="utf-8") as f:
            f.write(conc_text+"\n")
        
        
        # Save to BytesIO buffer
        buffered = io.BytesIO()
        annotated_pil.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()

        # Convert to base64 string
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        # Add prefix to make it usable in <img src="..." />
        img_base64 = f"data:image/jpeg;base64,{img_base64}"

        return {"status": "success", "annotated_image": img_base64,"license_plate":[conc_text]}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))