import cv2,easyocr,sys,os
from ultralytics import YOLO
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import gaussian
from math import sqrt 

class ImageProcessor:
    def __init__(self,frame_np,model_path):
        self.model = YOLO(model_path)
        self.frame_np = frame_np
        self.results= None
      
      
    def process_frame(self):
        self.results = self.model.track (
            source= self.frame_np,
            conf= 0.6,
            project= "outputs",
            name="frame",
            exist_ok = True
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
    
    def binary_threshold_inv(self,image,threshold=100,max_value=255):
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