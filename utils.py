def get_xyxy(results):
    """ 
        This function goes through each frame of the tracked video and returns a dictionary consisting
        of unique id of the detected license plate as key and a list of coordinates of the license plate
        along with the frame number in which the license plate was detected ie [x1,y1,x2,y2,frame_number].
    """
    frame_number=-1
    plate_dict={}

    for result in results:
        frame_number+=1
        for x in result.boxes.data.tolist():
            x1,y1,x2,y2,id,score,class_id=x

            if id not in plate_dict:
                plate_dict[int(id)]=[int(x1),int(y1),int(x2),int(y2),frame_number] 
                
    return plate_dict