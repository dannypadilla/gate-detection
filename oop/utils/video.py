'''
1. This program demonstrates some image/video PREPROCESSING
  for gate detection using opencv:
* # color filtering
* # convert to grayscale
* # use thresholds
* # find contours
* # draw boxes around objects
2. Outputs to a video file

Usage:
  video-gate.py [<args>]
   *args can be camera source/image-path/video-path

Output:
  Video is created in current directory
'''

import cv2
import numpy as np
import sys


# ************************* METHODS ****************************** #

# resize image to scale value param
def resize(frame, scale):
    return cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale) ))

# returns new scale value
def scaled(frame, scale):
    frame_shape_x, frame_shape_y, channels = frame.shape
    if(frame_shape_x > scale):
        return scale / frame_shape_x
    else:
        return 1

# filter out colors
def preprocess(frame, lower_upper_list):
    lower = np.array(lower_upper_list[0], dtype="uint8")
    upper = np.array(lower_upper_list[1], dtype="uint8")
    mask = cv2.inRange(frame, lower, upper)
    output = cv2.bitwise_and(frame, frame, mask = mask)
    return output, mask

# creates bounding boxes using each contour
def create_all_boxes(frame_contours_list):
    box_list = []
    for contour in frame_contours_list:
        cv2.boudingRect(contour)
        box_list.append(contour)
    return box_list

# filters the contour boxes based on size
def filter_boxes(frame_rectangle_list, filter_size=0):
    filtered_boxes = []
    for rectangle in frame_rectangle_list:
        if(rectangle[2] * rectangle[3] > filter_size):
            filtered_boxes.append(rectangle)
    return filtered_boxes

# draws GREEN boxes on image using contours boxes
def draw_rectangles(frame, frame_rectangle_list, x_offset=0, y_offset=0):
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
        source = sys.argv[1] # reads in source
    except:
        source = "img-to-video/run3.avi"


    ''' +++++++++++++++++++++ SETUP +++++++++++++++++++++ '''
    video = cv2.VideoCapture(source)
    
    # video attributes
    fps = 3.0 # fps
    frame_size = (744, 480) # has to be frame size of img
    new_file_name = "./run3_detection.avi" # output file name
    fourcc  = cv2.VideoWriter_fourcc(*"M", "J", "P", "G") # write object for mac
    out = cv2.VideoWriter(new_file_name, fourcc, fps, frame_size ) # output video writer

    # preprocess values
    lower_blue = np.array([55, 55, 55]) # lower bound
    upper_blue = np.array([150, 255, 255]) # upper bound
    #threshold_color = [0, 255, 0] # green - color of threshold
    box_filter_size = 400
    ''' ------------------- END-SETUP ------------------- '''

    
    while(video.isOpened() ):
        ret, frame = video1.read()

        if(ret):
            # PREPROCESS
            video_frame, mask = preprocess(frame, [lower_blue, upper_blue]) # color filter
            video_frame_gray = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY) # to grayscale
            ret, frame_thresh = cv2.threshold(video_frame_gray, 127, 255, cv2.THRESH_TOZERO) # get threshold
            frame_c, frame_contours, frame_heirarchy = cv2.findContours(frame_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # get contours

            # # # TEST # # #
            #frame_copy = frame.copy()
            #cv2.drawContours(frame_copy, frame_contours, -1, threshold_color, 3)

            frame_all_boxes = [cv2.boundingRect(contour) for contour in frame_contours] # get boxes
            frame_filtered_boxes = filter_boxes(frame_all_boxes, box_filter_size) # filter boxes
            draw_rectangles(frame, frame_filtered_boxes, 5, 5) # draw boxes
            out.write(frame) # write to file
            cv2.imshow("Run 3", frame)

            if(cv2.waitKey(1) & 0xFF == ord("q") ): # q to quit
                break
        else: # if end of video/camera-feed
            break

    out.release()
    video.release()
    cv2.destroyAllWindows()
