from moviepy import VideoFileClip
import os
from math import exp

def get_xyxy(results):
    """ 
        This function goes through each frame of the tracked video and returns a dictionary consisting
        of unique id of the detected license plate as key and a list of coordinates of the license plate
        along with the frame number in which the license plate was detected ie [x1,y1,x2,y2,frame_number].
    """
    threshold=0.6
    frame_number=-1
    plate_dict={}

    for result in results:
        frame_number+=1
        for x in result.boxes.data.tolist():
            if(len(x)==7):
                x1,y1,x2,y2,id,score,class_id=x

                if score > threshold and id not in plate_dict:
                    plate_dict[int(id)]=[int(x1),int(y1),int(x2),int(y2),frame_number]           
    return plate_dict


def convert_yolo_output_avi_to_mp4(project_dir: str, name: str, fileName: str) -> str:
    """
    Convert YOLO output avi video to mp4.
    `project_dir` is e.g. "outputs"
    `name` is e.g. "track" (folder inside project_dir)
    """
    avi_path = os.path.join(project_dir, name, fileName)
    # If YOLO names it differently, you can list files and find .avi file in that folder
    
    if not os.path.exists(avi_path):
        print(f"YOLO output video not found at {avi_path}")
        return None

    mp4_path = avi_path.replace(".avi", ".mp4")

    try:
        clip = VideoFileClip(avi_path)
        clip.write_videofile(mp4_path, codec='libx264', audio_codec='aac')
        clip.close()
        os.remove(avi_path)
        print(mp4_path)
    except Exception as e:
        print("Conversion failed:", e)
        return None
    return mp4_path


#def gaussian(x, sigma):
#    return exp(-(x ** 2) / (2 * sigma ** 2))