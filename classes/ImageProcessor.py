import cv2,easyocr,sys,os
from ultralytics import YOLO
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import gaussian
from math import sqrt 

class ImageProcessor:
    def __init__(self,frame_np,model_path):
        self.model=YOLO(model_path)
        self.frame_np=frame_np #np array
        self.results=None
      
        
    def process_frame(self):
        self.results=self.model.track(
            source=self.frame_np,
            conf=0.6,
            project="outputs",
            name="track",
            exist_ok=True
        )
        
    def annotated_image(self):
        return self.results[0].plot()
   
    def crop_license_plate(self):
        boxes = self.results[0].boxes.xyxy.numpy()
        x1, y1, x2, y2 = map(int, boxes[0])
        return self.frame_np[y1:y2, x1:x2]  
    
    def BGR2GRAY(self,lp):
        height,width,_ =lp.shape
       
        gray_img = np.zeros((height,width),dtype=np.uint8)
       
        for i in range(height):
            for j in range(width):
                B, G, R = lp[i,j]
                gray_value = int(0.114* B+ 0.587* G +0.288*R)
                gray_img[i,j] = gray_value
        return gray_img
    
    def bilateral_filter_gray(self, gray_image, diameter, sigma_spatial, sigma_range):
        padded_img = cv2.copyMakeBorder(gray_image, diameter//2, diameter//2, diameter//2, diameter//2, cv2.BORDER_REFLECT)
        filtered_img = np.zeros_like(gray_image, dtype=np.float64)
        rows, cols= gray_image.shape

        for i in range(rows):
            for j in range(cols):
                wp_total = 0
                pixel_val = 0 
                center_val = padded_img[i + diameter//2, j + diameter//2]
                
                for k in range(-diameter//2, diameter//2 + 1):
                    for l in range(-diameter//2, diameter//2 + 1):
                        neighbor_val = padded_img[i + k + diameter//2, j + l + diameter//2]

                        gs = gaussian(sqrt(k**2 + l**2), sigma_spatial)
                        gr = gaussian(abs(int(center_val) - int(neighbor_val)), sigma_range)
                        w = gs * gr

                        pixel_val += neighbor_val * w
                        wp_total += w
                        

                filtered_img[i, j] = pixel_val / wp_total

        return filtered_img.astype(np.uint8)
    
    def binary_threshold_inv(self,image,threshold=200,max_value=255):
        result = np.zeros_like(image)
        
        height,width=image.shape
        
        for i in range(height):
            for j in range(width):
                pixel = image[i, j]
                if pixel > threshold:
                    result[i, j] = 0
                else:
                    result[i, j] = max_value    
                    
        return result
        
    def erode(self, image, kernel):
        m, n = kernel.shape
        pad_m, pad_n = m//2, n//2
        padded = np.pad(image, ((pad_m, pad_m), (pad_n, pad_n)), constant_values=0)
        out = np.zeros_like(image)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i+m, j:j+n]
                if np.all(region[kernel==1] == 255):  # 255 = foreground
                    out[i, j] = 255
        return out

    def dilate(self, image, kernel):
        m, n = kernel.shape
        pad_m, pad_n = m//2, n//2
        padded = np.pad(image, ((pad_m, pad_m), (pad_n, pad_n)), constant_values=0)
        out = np.zeros_like(image)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i+m, j:j+n]
                if np.any(region[kernel==1] == 255):
                    out[i, j] = 255
        return out

    def opening(self, image, kernel):
        eroded = self.erode(image, kernel)
        opened = self.dilate(eroded, kernel)
        return opened

    def closing(self, image, kernel):
        dilated = self.dilate(image, kernel)
        closed = self.erode(dilated, kernel)
        return closed
    
    
# imageProcessor=ImageProcessor("viber_image_2025-07-27_23-41-23-168.jpg","../runs/detect/train/weights/best.pt")

# imageProcessor.process_frame()

# detected_frame=imageProcessor.annotated_image()
# resized_frame = cv2.resize(detected_frame, (800, 600))
# cv2.imshow('annotated image',resized_frame)
# cv2.waitKey(0)              
# cv2.destroyAllWindows() 

# lp=imageProcessor.crop_license_plate()
# cv2.imshow("plate",lp)
# cv2.waitKey(0)              
# cv2.destroyAllWindows() 

# resized = cv2.resize(lp, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
# cv2.imshow("resized",resized)
# cv2.waitKey(0)              
# cv2.destroyAllWindows() 

# #convert from BGR to Grayscale
# lp_gray=imageProcessor.BGR2GRAY(resized)
# cv2.imshow("grayplate",lp_gray)
# cv2.waitKey(0)              
# cv2.destroyAllWindows() 

# #Denoising
# # denoised = cv2.bilateralFilter(lp_gray, 5, 5, 5)
# denoised = imageProcessor.bilateral_filter_gray(lp_gray, 5, 5, 5)
# cv2.imshow("denoised_plate",denoised)
# cv2.waitKey(0)              
# cv2.destroyAllWindows()

# #apply thresholding
# lp_thresh=imageProcessor.binary_threshold_inv(denoised)
# cv2.imshow("lp_thresh_plate",lp_thresh)
# cv2.waitKey(0)              
# cv2.destroyAllWindows() 

# #morphological transformation
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
# morphed = cv2.morphologyEx(lp_thresh, cv2.MORPH_OPEN, kernel)
# # cv2.imshow("plate",morphed)
# # cv2.waitKey(0)              
# # cv2.destroyAllWindows() 

# texts = imageProcessor.reader.readtext(morphed,allowlist='०१२३४५६७८९कखगघङचछजझञटठडढणतथधनपफबभमयरलवशषसहक्षत्रज्ञािीुूेैोौंःँ -.')

# for text in texts:
#     _, text, text_score = text
#     print(text,text_score)
#     with open("outputs/track/plate_text.txt", "a", encoding="utf-8") as f:
#             f.write(text+"\n")