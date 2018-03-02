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

def resize(frame, scale): # resize image to scale value param
    return cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale) ))

def scaled(frame, scale): # returns new scale value
    frame_shape_x, frame_shape_y, channels = frame.shape
    if(frame_shape_x > scale):
        return scale / frame_shape_x
    else:
        return 1

## -- preprocess utils -- ##

def preprocess(frame, lower_upper_list):
    lower = np.array(lower_upper_list[0], dtype="uint8")
    upper = np.array(lower_upper_list[1], dtype="uint8")

    mask = cv2.inRange(frame, lower, upper)
    output = cv2.bitwise_and(frame, frame, mask = mask)

    return output, mask

def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR3GRAY)

def apply_threshold(img):
    ret, img_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO) ## fixed for now
    return (ret, img_thresh)

def find_contours(img):
    img, img_contours, img_heirarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return (img, img_contours, img_heirarchy)

def filter_contours(contours):
    new_contours_list = []
    for cont in contours:
        if len(cont) > lower_bound and len(cont) < upper_bound:
            new_contours_list.append(cont)
    return np.array(new_contour_list)

## -- end preprocess utils -- ##

## -- drawing utils -- ##

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

## -- end drawing utils -- ##

## -- directory utils -- ##

def get_dir_list(imgs_path):
    dir_list = []
    with os.scandir(imgs_path) as it:
        for entry in it:
            if not entry.name.startswith(".") and entry.is_file:
                dir_list.append(imgs + path + entry.name)
        return dir_list

def get_img_data(imgs_path):
    img_list = []
    for img in imgs_path:
        img_list.append(cv2.imgread(img) )
    return img_list

def get_features_with_label(image_data, hog, dimensions, label):
    data = []
    for img in image_data:
        image = cv2.resize(img, dimensions)
        feats = hog.compute(image[:, :, 2] ) ## CHECK THIS - s/b 3 channel
        data.append((feats, label) )
    return data

## --- end directory utils --- ##

## unsupervised LATER

## --- classification methods --- ##

def get_positive_roi(img, roi_list, x_offset=0, y_offset=0):
    positive_roi = []
    for x, y, w, h in roi_list:
        roi = img[y:y + h, x:x + w, :3] ## CHECK THIS] - 3 channels works :)
        roi_resized = cv2.resize(roi, dimensions)
        features = hog.compute(roi_resized)
        proba = svm.predict_proba(features.reshape(1, -1) )[0]
        if proba[1] > .1:
            positive_roi.append((x, y, w, h) )
    return positive_roi

## -- video utils -- ##

def create_video_writer():
    pass

## -- end video utils -- ##

# **************************************************************** #

if __name__ == '__main__':

    print(__doc__)

    try:
        fn = sys.argv[1] # unused for now
    except:
        fn = 0

    ## these will eventually become cmdline args
    video1_path = "img-to-video/run3.avi"
    video2_path = "img-to-video/run4.avi"
    
    video1 = cv2.VideoCapture(video2_path)
    
    lower_blue = np.array([55, 55, 55])
    upper_blue = np.array([150, 255, 255])
    threshold_color = [0, 255, 0] # green
    box_filter_size = 400

    ## for outputting video
    fps = 3.0
    file_name = "./run3_.avi"
    # create write object
    fourcc  = cv2.VideoWriter_fourcc(*"M", "J", "P", "G") # for mac
    out = cv2.VideoWriter(file_name, fourcc, fps, (744, 480) ) # has to be frame size of img

    while(video1.isOpened() ):
        ret, frame = video1.read()

        if(ret):
            video_frame_1, mask = preprocess(frame, [lower_blue, upper_blue]) # preprocess
            video_frame_gray = cv2.cvtColor(video_frame_1, cv2.COLOR_BGR2GRAY) # gray
            ret, frame_thresh = cv2.threshold(video_frame_gray, 127, 255, cv2.THRESH_TOZERO)
            frame_c, frame_contours, frame_heirarchy = cv2.findContours(frame_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            #frame_copy = frame.copy()
            #cv2.drawContours(frame_copy, frame_contours, -1, threshold_color, 3)

            frame_all_boxes = [cv2.boundingRect(c) for c in frame_contours]
            frame_filtered_boxes = filter_boxes(frame_all_boxes, box_filter_size)
            draw_rectangles(frame, frame_filtered_boxes, 5, 5) # last 2 params are offset

            # write to file
            out.write(frame)

            cv2.imshow("Run 3", frame)

            if(cv2.waitKey(1) & 0xFF == ord("q") ):
                break
        else:
            break

    out.release()
    video1.release()
    cv2.destroyAllWindows()
