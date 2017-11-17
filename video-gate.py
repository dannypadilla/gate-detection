import cv2
import numpy as np
from matplotlib import pyplot as plt

def resize(img, scale): # resize image to scale value param
    return cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale) ))

def scaled(img, scale): # returns new scale value
    img_shape_x, img_shape_y, channels = img.shape
    if(img_shape_x > scale):
        return scale / img_shape_x
    else:
        return 1

def preprocess(image, lower_upper_list):
    lower = np.array(lower_upper_list[0], dtype="uint8")
    upper = np.array(lower_upper_list[1], dtype="uint8")

    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask = mask)

    return output, mask

video1_path = "img-to-video/run3.avi"
video2_path = "img-to-video/run4.avi"

video1 = cv2.VideoCapture(video1_path)

while(video1.isOpened() ):
    
    ret, frame = video1.read()
    if(ret):
        cv2.imshow("Run 3", frame)
        if(cv2.waitKey(1) & 0xFF == ord("q") ):
            break
    else:
        break

video1.release()
cv2.destroyAllWindows()
