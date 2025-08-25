from fastapi import FastAPI, HTTPException
import base64,io
from io import BytesIO
from PIL import Image
import numpy as np
from models import ImageInput
from fastapi.middleware.cors import CORSMiddleware
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from classes.ImageProcessor import ImageProcessor
import cv2,easyocr

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
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        imageProcessor=ImageProcessor(image_bgr,"../runs/detect/train/weights/best.pt")
        imageProcessor.process_frame()

        annotated_img=imageProcessor.annotated_image()
        annotated_img_rgb=cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        lp=imageProcessor.crop_license_plate()
        cv2.imshow("plate",lp)
        cv2.waitKey(0)              
        cv2.destroyAllWindows() 
        
        annotated_pil = Image.fromarray(annotated_img_rgb)
        
        resized = cv2.resize(lp, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("resized",resized)
        cv2.waitKey(0)              
        cv2.destroyAllWindows() 
        cv2.imwrite("outputs/track/cropped_plate.jpg", lp)
        
        lp_gray=imageProcessor.BGR2GRAY(resized)
        
        denoised = imageProcessor.bilateral_filter_gray(lp_gray, 3, 3, 3)
        cv2.imshow("denoised_plate",denoised)
        cv2.waitKey(0)              
        cv2.destroyAllWindows()
        
        lp_thresh=imageProcessor.binary_threshold_inv(denoised)
        cv2.imshow("lp_thresh_plate",lp_thresh)
        cv2.waitKey(0)              
        cv2.destroyAllWindows()

        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # morphed = cv2.morphologyEx(lp_thresh, cv2.MORPH_OPEN, kernel)
        
        kernel = np.ones((5,5), dtype=np.uint8)
        opened = imageProcessor.opening(lp_thresh, kernel)
        cv2.imshow("opened",opened)
        cv2.waitKey(0)              
        cv2.destroyAllWindows()
        
        closed = imageProcessor.closing(opened, kernel)
        cv2.imshow("closed",closed)
        cv2.waitKey(0)              
        cv2.destroyAllWindows()
        
        ocr_results = reader.readtext(closed,allowlist='०१२३४५६७८९कखगघङचछजझञटठडढणतथधनपफबभमयरलवशषसहक्षत्रज्ञािीुूेैोौंःँ -.')
        print(ocr_results)
        texts = [res[1] for res in ocr_results]
        conc_text = ""
    
        for text in texts:
            conc_text+=text
        
        print("LP: " +conc_text)
        
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