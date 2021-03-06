'''
1. This program demonstrates some image/video preprocessing
  for gate detection using opencv:
* # color filtering
* # convert to grayscale
* # use thresholds
* # find contours
* # draw boxes around objects
2. Outputs to a video file too

Usage:
  video-gate.py [<args_tbd>]

  Output is created in current directory
'''

import cv2
import numpy as np
import sys
from sklearn.externals import joblib

def resize(frame, scale): # resize image to scale value param
    return cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale) ))

def scaled(frame, scale): # returns new scale value
    frame_shape_x, frame_shape_y, channels = frame.shape
    if(frame_shape_x > scale):
        return scale / frame_shape_x
    else:
        return 1

def preprocess(frame, lower_upper_list):
    lower = np.array(lower_upper_list[0], dtype="uint8")
    upper = np.array(lower_upper_list[1], dtype="uint8")

    mask = cv2.inRange(frame, lower, upper)
    output = cv2.bitwise_and(frame, frame, mask = mask)

    return output, mask

def create_all_boxes(frame_contours_list):
    box_list = []
    for contour in frame_contours_list:
        cv2.boudingRect(contour)
        box_list.append(contour)
    return box_list

def filter_boxes(frame_rectangle_list, filter_size=0):
    filtered_boxes = []
    for rectangle in frame_rectangle_list:
        if(rectangle[2] * rectangle[3] > filter_size):
            filtered_boxes.append(rectangle)
    return filtered_boxes

def draw_rectangles(frame, frame_rectangle_list, x_offset = 0, y_offset = 0):
    for x, y, w, h in frame_rectangle_list:
        cv2.rectangle(
            frame,
            (x - x_offset, y - y_offset),
            ((x + x_offset) + w, (y + y_offset) + h),
            (0, 255, 0),
            2
        )

# **************************************************************** #

if __name__ == '__main__':

    print(__doc__)

    try:
        fn = sys.argv[1] # unused for now
    except:
        fn = 0

    ## these will eventually become cmdline args
    video_path = "videos/gate_new.avi"
    video = cv2.VideoCapture(video_path)
    lower_blue = np.array([55, 55, 55])
    upper_blue = np.array([150, 255, 255])
    threshold_color = [0, 255, 0] # green
    box_filter_size = 400

    # model/descriptor
    svm = None
    model_name = "py3_svm.pkl"
    try:
        svm = joblib.load("models/gate/" + model_name)
        print("Loading model...")
    except:
        print("No model available...")
    min_dims = 80
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    bins = 9
    dims = (80, 80)
    hog = cv2.HOGDescriptor(dims, block_size, block_stride, cell_size, bins)

    ## for outputting video
    fps = 30.0
    file_name = "./run3_.avi"
    # create write object
    fourcc  = cv2.VideoWriter_fourcc(*"M", "J", "P", "G") # for mac
    #out = cv2.VideoWriter(file_name, fourcc, fps, (744, 480) ) # has to be frame size of img
    out = cv2.VideoWriter(file_name, fourcc, fps, (640, 480) ) # has to be frame size of img

    while(video.isOpened() ):
        ret, frame = video.read()

        if(ret):
            rows, cols,_ = frame.shape
            rot_trans = cv2.getRotationMatrix2D( (cols/2, rows/2), 180, 1) # rotate image 180
            frame = cv2.warpAffine(frame, rot_trans, (cols, rows) ) # since camera is upside down..
            
            video_frame, mask = preprocess(frame, [lower_blue, upper_blue]) # preprocess
            video_frame_gray = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY) # gray
            ret, frame_thresh = cv2.threshold(video_frame_gray, 127, 255, cv2.THRESH_TOZERO)
            frame_c, frame_contours, frame_heirarchy = cv2.findContours(frame_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            #frame_copy = frame.copy()
            #cv2.drawContours(frame_copy, frame_contours, -1, threshold_color, 3)

            # contours length filter was here -- RIP

            frame_all_boxes = [cv2.boundingRect(c) for c in frame_contours]
            frame_filtered_boxes = filter_boxes(frame_all_boxes, box_filter_size)
            draw_rectangles(frame, frame_filtered_boxes, 5, 5) # last 2 params are offset

            # write to file
            out.write(frame)

            cv2.imshow("Gate", frame)

            if(cv2.waitKey(1) & 0xFF == ord("q") ):
                break
        else:
            break

    out.release()
    video.release()
    cv2.destroyAllWindows()
