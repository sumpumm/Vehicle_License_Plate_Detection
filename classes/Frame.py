import cv2,easyocr,os

class Frame:
    def __init__(self):
        self.reader = easyocr.Reader(['ne'])

    def extract_plate(self, frame, coords, id):
        x1, y1, x2, y2 = coords
        crop = frame[y1:y2, x1:x2]
        ocr_results = self.reader.readtext(crop)

        
        os.makedirs("outputs/extracted_plate", exist_ok=True)
        cv2.imwrite(f"outputs/extracted_plate/plate_{id}.jpg", crop)

        
        texts = [res[1] for res in ocr_results]
        return texts
