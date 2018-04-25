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
import pandas as pd
import sys
import glob
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# **************************** USEFUL METHODS ********************* #

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


def get_features_with_label(img_data, hog, label):
    dims = (80, 80)
    data = []
    for img in img_data:
        img = cv2.resize(img, dims)
        feat = hog.compute( img[:, :, :] )
        data.append( (feat, label) )
    return data


def train_svm(svm, hog, positive_images_path, negative_images_path):
    pos_imgs = []
    neg_imgs = []
    for img in glob.glob(positive_images_path):
        pos_imgs.append( cv2.imread(img) )
    for img in glob.glob(negative_images_path):
        neg_imgs.append( cv2.imread(img) )

    positive_data = get_features_with_label(pos_imgs, hog, 1)
    negative_data = get_features_with_label(neg_imgs, hog, 0)
    data_df = positive_data + negative_data
    np.random.shuffle(data_df)

    feat, labels = map(list, zip(*data_df) )
    feat_flat = [x.flatten() for x in feat]
    X_df = pd.DataFrame(feat_flat)
    y_df = pd.Series(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X_df,
        y_df,
        test_size=0.3,
        random_state=2
    )
    svm.fit(X_train, y_train)


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


def draw_rectangles(frame, frame_rectangle_list, color, x_offset=0, y_offset=0):
    for x, y, w, h in frame_rectangle_list:
        cv2.rectangle(
            frame,
            (x - x_offset, y - y_offset),
            ((x + x_offset) + w, (y + y_offset) + h),
            color,
            2
        )

# ****************************** /END METHODS/ ******************** #

# ***************************************************************** #

if __name__ == '__main__':

    print(__doc__)

    try:
        fn = sys.argv[1] # unused for now
    except:
        fn = 0

    # ****************************** /IMAGES SETUP/ ************************ #

    video_dict = {
        1: "gate_jon_1.avi",
        2: "gate_jon_2.avi",
        3: "gate_jon_3.avi",
        4: "no_gate_@5fps.avi",
        5: "old_run4_@3fps.avi"
    }

    pos_img_dict = {
        1: "images/whole_gate/*.jpg",
        2: "images/bars/*.jpg",
        3: "images/whole_gate_and_bars/*.jpg",
        4: "jupyter/positive/*.jpg"
    }

    neg_img_dict = {
        1: "images/negatives/*.jpg",
        2: "jupyter/negative/*.jpg"
    }
    
    vid = 5
    pos = 1
    neg = 1
    video_path = "videos/" + video_dict[vid]
    positive_images_path = pos_img_dict[pos]
    negative_images_path = neg_img_dict[neg]

    svm_choices = str(pos) + str(neg) # numbers correspond to dict values used
    choices = str(vid) + str(pos) + str(neg) # numbers correspond to dict values used
    model_name = "svm_" + svm_choices

    # ****************************** /END IMAGES SETUP/ ******************** #

    ## these will eventually become cmdline args
    #video_path = "videos/gate_new.avi"
    video = cv2.VideoCapture(video_path)
    lower_blue = np.array([0, 50, 50])
    upper_blue = np.array([130, 250, 255])
    threshold_color = [0, 255, 0] # green
    box_filter_size = 400

    # model/descriptor
    min_dims = 80
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    bins = 9
    dims = (80, 80)
    hog = cv2.HOGDescriptor(dims, block_size, block_stride, cell_size, bins) # HOG
    
    svm = None # SVM
    model_path = "models/gate/"
    #model_name = "orig_svm" # so I can append to video file name too..
    model_file_name = model_name + ".pkl" # whole file name
    vers_label = "py3"
    path = model_path + vers_label + "_" + model_file_name

    try:
        svm = joblib.load(path)
        print("\nLoading model from disk...\n")
    except:
        print("\nTraining model...")
        svm = SVC(C=1.0, kernel="linear", probability=True, random_state=2)
        #svm = SVR(C=1.0, kernel="linear")
        train_svm(svm, hog, positive_images_path, negative_images_path)
        joblib.dump(svm, path) # store model object to disk
        print("\nStoring model to location: " + "\"" + path + "\"\n")
        
    ## for outputting video
    fps = 8.0
    #file_name = "./run_jons_.avi"
    file_name = "./gate_" + choices + ".avi"
    fourcc  = cv2.VideoWriter_fourcc(*"M", "J", "P", "G") # create write object for mac

    # since the videos res and orientation are different
    camera_is_upside_down = False
    if(vid <= 3):
        out = cv2.VideoWriter(file_name, fourcc, fps, (640, 480) ) # has to be frame size of img
        camera_is_upside_down = True
    else:
        out = cv2.VideoWriter(file_name, fourcc, fps, (744, 480) ) # has to be frame size of img

    # start video processing
    while(video.isOpened() ):
        ret, frame = video.read()

        if(ret):
            
            if(camera_is_upside_down): # whether camera should be rotated 180 deg
                rows, cols, ch = frame.shape
                rot_trans = cv2.getRotationMatrix2D( (cols/2, rows/2), 180, 1) # rotate image 180
                frame = cv2.warpAffine(frame, rot_trans, (cols, rows) ) # since camera is upside down..
            
            #video_frame, mask = preprocess(frame, [lower_blue, upper_blue]) # preprocess
            #video_frame_gray = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY) # gray

            ''' TESTING ''' 
            frame_b = frame.copy()
            frame_g = frame.copy()
            frame_r = frame.copy()
            frame_b[:,:,0] = 255
            frame_g[:,:,1] = 255
            frame_r[:,:,2] = 255
            frame_b_gray = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
            frame_g_gray = cv2.cvtColor(frame_g, cv2.COLOR_BGR2GRAY)
            frame_r_gray = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

            ## @all0 -   r-b=X, b-r=~~X, r-g=X,  g-r=X,   b-g=test, g-b=~X     --> THRESH
            ## @all0 -   r-b=X, b-r=X,   r-g=~X, g-r=X,   b-g=X,    g-b=test   --> THRESH_INV
            ## @all255 - r-b=X, b-r=X,   r-g=~X, g-r=~ok, b-g=~ok,  g-b=better --> THRESH
            ## @all255 - r-b=X, b-r=X,   r-g=~X, g-r=~X,  b-g=~ok,  g-b=~ok    --> THRESH_INV

            video_frame_gray = frame_g_gray - frame_b_gray # works - @all255
            #video_frame_gray = frame_r_gray - frame_b_gray # testing
            ''' END TESTING '''
            
            ret, frame_thresh = cv2.threshold(video_frame_gray, 127, 255, cv2.THRESH_TOZERO)
            #ret, frame_thresh = cv2.threshold(video_frame_gray, 127, 255, cv2.THRESH_TOZERO_INV)
            frame_c, frame_contours, frame_heirarchy = cv2.findContours(frame_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # contours length filter was here -- RIP
            min_cont_size = 100
            max_cont_size = 5000
            new_cont_list = []
            for cont in frame_contours:
                cont_len = len(cont)
                if ( (cont_len > min_cont_size) and (cont_len < max_cont_size) ):
                    new_cont_list.append(cont)
            filtered_contours = np.array(new_cont_list)

            #frame_copy = frame.copy()
            #cv2.drawContours(frame_copy, new_cont_list, -1, threshold_color, 3)

            #frame_all_boxes = [cv2.boundingRect(c) for c in frame_contours]
            frame_all_boxes = [cv2.boundingRect(c) for c in filtered_contours]
            frame_filtered_boxes = filter_boxes(frame_all_boxes, box_filter_size)

            all_cont_color = [0, 0, 255] # red
            positive_roi = []
            dimensions = (80, 80)
            for x, y, w, h in frame_filtered_boxes:
                roi = frame[y:y + h, x:x + w, :]
                roi_resized = cv2.resize(roi, dimensions) # dimensions defined as (80, 80) above
                features = hog.compute(roi_resized)
                feat_reshape = features.reshape(1, -1)
                proba = svm.predict_proba(feat_reshape)[0] # [0] since returns a 2d array.. [[x]]
                prediction = svm.predict(feat_reshape) # 0 or 1
                gate_class = proba[1] # corresponds to class 1 (positive gate)
                if prediction > 0 and gate_class >= .9:
                    positive_roi = [(x, y, w, h)]
                    #positive_roi.append( (x, y, w, h) )
                    print("\nprediction %", gate_class, "\n")
            draw_rectangles(frame, positive_roi, threshold_color, 5, 5) # last 2 params are offset
            #draw_rectangles(frame, frame_filtered_boxes, all_cont_color, 5, 5) # last 2 params are offset

            # write to file
            out.write(frame)

            ''' VIEW MULTIPLE TEST SCREENS '''
            cv2.imshow("gate", frame) # actual frame
            #cv2.resizeWindow("Gate", 100, 100)
            cv2.moveWindow("gate", 0, 0)
            
            cv2.imshow("thresholding", frame_thresh) # threshold frame
            cv2.moveWindow("thresholding", 0, 500)

            cv2.imshow("grayscale", video_frame_gray) # grayscale
            cv2.moveWindow("grayscale", 500, 0)

            #cv2.imshow("contours", frame_copy) # contours
            #cv2.moveWindow("contours", 500, 500)

            if(cv2.waitKey(1) & 0xFF == ord("q") ):
                break
        else:
            break

    #print(svm.classes_)
    out.release()
    video.release()
    cv2.destroyAllWindows()