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
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

from lists.video_list import video_dict
from lists.training_data_list import pos_img_dict, neg_img_dict

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


def filter_contours(frame_countours, min_cont_size=100, max_cont_size=5000):
    new_cont_list = []
    for cont in frame_contours:
        cont_len = len(cont)
        if ( (cont_len > min_cont_size) and (cont_len < max_cont_size) ):
            new_cont_list.append(cont)
    filtered_contours = np.array(new_cont_list)
    return filtered_contours


## made to declutter main()
def color_subtract_test():
    ''' TESTING - COLOR SUBTRACTION '''
    frame_b = frame.copy()
    frame_g = frame.copy()
    frame_r = frame.copy()
    frame_b[:,:,0] = 255
    frame_g[:,:,1] = 255
    frame_r[:,:,2] = 255
    frame_b_gray = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    frame_g_gray = cv2.cvtColor(frame_g, cv2.COLOR_BGR2GRAY)
    frame_r_gray = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

            ## NOTES (IGNORE PLZ)
            ## @all0 -   r-b=X, b-r=~~X, r-g=X,  g-r=X,   b-g=test, g-b=~X     --> THRESH
            ## @all0 -   r-b=X, b-r=X,   r-g=~X, g-r=X,   b-g=X,    g-b=test   --> THRESH_INV
            ## @all255 - r-b=X, b-r=X,   r-g=~X, g-r=~ok, b-g=~ok,  g-b=better --> THRESH
            ## @all255 - r-b=X, b-r=X,   r-g=~X, g-r=~X,  b-g=~ok,  g-b=~ok    --> THRESH_INV

            #video_frame_gray = frame_g_gray - frame_b_gray # works - @all255
            #video_frame_gray = frame_r_gray - frame_b_gray # testing
    ''' END TESTING '''
    print("END OF COLOR_SUBTRACT_TEST")


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


def filter_boxes(frame_rectangle_list, filter_size=4000):
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

# reshape an ROI to a square image
# tuple in, tuple out
def roi_to_square(img, x, y, w, h):
    frame_y, frame_x, ch = img.shape
    x_shape = w # size of x ROI
    y_shape = h # size of y ROI
    
    if(x_shape > y_shape):
        y_diff = (x_shape - y_shape) // 2
        if( (y - y_diff) < 0):
            return (x, y, w, h + (y_diff * 2) ) # add only to bottom
        elif( (h + y_diff) > frame_y):
            return (x, y - (y_diff * 2), w, h) # add only to top
        else:
            return (x, y - y_diff, w, h + y_diff) # add to both top and bottom
    elif(y_shape > x_shape):
        x_diff = (y_shape - x_shape) // 2
        if( (x - x_diff) < 0):
            return (x, y, w + (x_diff * 2), h) # add only to right side
        elif( (w + x_diff) > frame_x):
            return (x - (x_diff * 2), y, w, h) # add only to left side
        else:
            return (x - x_diff, y, w + x_diff, h) # add to left and right side
    else:
        return (x, y, w, h) # already square


# ****************************** /END METHODS/ ******************** #

# ***************************************************************** #

if __name__ == '__main__':

    print(__doc__)

    try:
        fn = sys.argv[1] # unused for now
    except:
        fn = 0

    # ****************************** /IMAGES SETUP/ ************************ #

    vid = 90 # 30 - 56: 7/5/16 video sets *** 57 - 73 7/16/18
    pos = 16 # 3
    neg = 10 # 1
    video_path = "videos/" + video_dict[vid]
    positive_images_path = pos_img_dict[pos]
    negative_images_path = neg_img_dict[neg]
    choices = str(vid) + "-" + str(pos) + "-" + str(neg) # numbers correspond to dict values used - for naming

    # model setup
    min_prob = .1
    svm_choices = str(pos) + "-"  + str(neg) # numbers correspond to dict values used - for naming
    model_name = "svm_" + svm_choices

    # ****************************** /END IMAGES SETUP/ ******************** #

    ## these will eventually become cmdline args
    video = cv2.VideoCapture(video_path)

    # orig - good
    #lower_blue = np.array([0, 100, 100])
    #upper_blue = np.array([60, 255, 255])

    # new vals - slightly better, allows a little more 'black', testing is ok
    #lower_blue = np.array([0, 100, 50])
    #upper_blue = np.array([10, 255, 255])

    #lower_blue = np.array([0, 150, 0])
    #upper_blue = np.array([10, 255, 255])

    lower_blue = np.array([0, 100, 0])
    upper_blue = np.array([180, 200, 150])
    
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
    model_file_name = model_name + ".pkl" # whole file name
    vers_label = "py3"
    path = model_path + vers_label + "_" + model_file_name

    try:
        svm = joblib.load(path)
        print("\nLoading model from disk...\n")
    except:
        print("\nTraining model...")
        svm = SVC(C=0.1, kernel="linear", probability=True, random_state=2)
        train_svm(svm, hog, positive_images_path, negative_images_path)
        joblib.dump(svm, path) # store model object to disk
        print("\nStoring model to location: " + "\"" + path + "\"\n")
        
    ## for outputting video
    fps = 30.0 # 8.0 orig
    file_name = "./gate_" + choices + ".avi"
    fourcc  = cv2.VideoWriter_fourcc(*"M", "J", "P", "G") # create write object for mac

    # since the videos res and orientation are different
    camera_is_upside_down = False
    if(vid < 4 or vid > 5 or vid < 10):
        out = cv2.VideoWriter(file_name, fourcc, fps, (640, 480) ) # has to be frame size of img
        if(vid < 4):
            camera_is_upside_down = True
    else:
        out = cv2.VideoWriter(file_name, fourcc, fps, (744, 480) ) # has to be frame size of img
    

    ####### REMOVE!!! KLAJLKJDKFJKSLDF 
    out = cv2.VideoWriter(file_name, fourcc, fps, (744, 480) ) # has to be frame size of img
    #/#########
    
    predicted_counter = 0
    roi_counter = 1
    store_roi_to = "images/_rois/img_roi_ " + choices + "_" + str(min_prob) + "_" # for storing found ROI's to disk

    kernel = np.ones( (5, 5), np.uint8)
    
    # start video processing
    while(video.isOpened() ):
        ret, frame = video.read()

        if(ret):
            if(camera_is_upside_down): # whether camera should be rotated 180 deg
                rows, cols, ch = frame.shape
                rot_trans = cv2.getRotationMatrix2D( (cols/2, rows/2), 180, 1) # rotate image 180
                frame = cv2.warpAffine(frame, rot_trans, (cols, rows) ) # since camera is upside down..

            # blur before
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            ''' BLUR - FILTERING '''
            # OPTIONAL
            #frame_blur = cv2.bilateralFilter(frame_hsv, 9, 100, 100)
            #frame_blur = cv2.GaussianBlur(frame_hsv, (5, 5), 0)
            ''' END BLUR - FILTERING END'''

            # blur after
            #frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # choose between blur or non blur - OPTIONAL
            video_frame, mask = preprocess(frame_hsv, [lower_blue, upper_blue]) # preprocess
            #video_frame, mask = preprocess(frame_blur, [lower_blue, upper_blue]) # blur

            #video_frame_gray = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY) # gray

            #ret, frame_thresh = cv2.threshold(video_frame_gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # allow more

            close_frame = cv2.morphologyEx(video_frame, cv2.MORPH_CLOSE, kernel) # fill in
            #close_frame = cv2.morphologyEx(frame_thresh, cv2.MORPH_CLOSE, kernel) # fill in
            #open_frame = cv2.morphologyEx(close_frame, cv2.MORPH_OPEN, kernel) # remove specs
            dilate_frame = cv2.dilate(close_frame, kernel, iterations=3) # make chubby

            # to grayscale
            #vid_hsv2bgr = cv2.cvtColor(video_frame, cv2.COLOR_HSV2BGR)
            #video_frame_gray = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY) # gray
            
            vid_hsv2bgr = cv2.cvtColor(dilate_frame, cv2.COLOR_HSV2BGR)
            video_frame_gray = cv2.cvtColor(vid_hsv2bgr, cv2.COLOR_BGR2GRAY) # gray

            # thresholding - OPTIONAL
            #ret, frame_thresh = cv2.threshold(video_frame_gray, 127, 255, cv2.THRESH_TOZERO)
            ret, frame_thresh = cv2.threshold(video_frame_gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # allow more

            '''
            # noise removal
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(frame_thresh, cv2.MORPH_OPEN,kernel, iterations = 2)
            # sure background area
            sure_bg = cv2.dilate(opening,kernel,iterations=3)
            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
            ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            # Marker labelling
            ret, markers = cv2.connectedComponents(sure_fg)
            # Add one to all labels so that sure background is not 0, but 1
            markers = markers + 1
            # Now, mark the region of unknown with zero
            markers[unknown==255] = 0

            img_copy = frame.copy()
            markers = cv2.watershed(img_copy, markers)
            img_copy[markers == -1] = [255, 0, 0]

            ret, frame_thresh_wat = cv2.threshold(video_frame_gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # allow more
            '''

            ''' MORPH OPS
            kernel = np.ones( (5, 5), np.uint8) # tmp for now
            #erode_frame = cv2.erode(frame_thresh, kernel, iterations=1) # fade/trim
            #open_frame = cv2.morphologyEx(erode_frame, cv2.MORPH_OPEN, kernel) # remove specs
            open_frame = cv2.morphologyEx(frame_thresh, cv2.MORPH_OPEN, kernel) # remove specs
            close_frame = cv2.morphologyEx(open_frame, cv2.MORPH_CLOSE, kernel) # fill in
            dilate_frame = cv2.dilate(close_frame, kernel, iterations=1) # make chubby
            '''
            
            # find contours
            frame_c, frame_contours, frame_heirarchy = cv2.findContours(frame_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #frame_c, frame_contours, frame_heirarchy = cv2.findContours(dilate_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #frame_c, frame_contours, frame_heirarchy = cv2.findContours(frame_thresh_wat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # filter contours based on length - defaule min = 100, max = 5000
            filtered_contours = filter_contours(frame_contours, min_cont_size=100, max_cont_size=1000)

            # draw contours for visual
            frame_copy = frame.copy()
            cv2.drawContours(frame_copy, filtered_contours, -1, threshold_color, 3)

            # select between all contours and filtered
            #frame_all_boxes = [cv2.boundingRect(c) for c in frame_contours]
            frame_all_boxes = [cv2.boundingRect(c) for c in filtered_contours]
            
            frame_filtered_boxes = filter_boxes(frame_all_boxes, box_filter_size)

            all_cont_color = [0, 0, 255] # red
            positive_roi = []
            dimensions = (80, 80)
            max_val = 0
            
            for x, y, w, h in frame_filtered_boxes:
                #x_sq, y_sq, w_sq, h_sq = roi_to_square(frame, x, y, w, h)
                
                roi = frame[y:y + h, x:x + w, :]
                #roi = frame[y_sq:y_sq + h_sq, x_sq:x_sq + w_sq, :]
                
                roi_resized = cv2.resize(roi, dimensions) # dimensions defined as (80, 80) above
                #cv2.imwrite(store_roi_to + str(roi_counter) + "_resized_" + ".jpg", roi_resized) # save roi to disk
                features = hog.compute(roi_resized)
                feat_reshape = features.reshape(1, -1)
                proba = svm.predict_proba(feat_reshape)[0] # [0] since returns a 2d array.. [[x]]
                prediction = svm.predict(feat_reshape) # 0 or 1
                gate_class = proba[1] # corresponds to class 1 (positive gate)
                #cv2.imwrite(store_roi_to + str(roi_counter) + "_resized_" + str(prediction) + "_" + str(gate_class)  + ".jpg", roi_resized) # save roi to disk
                if prediction > 0 and gate_class >= min_prob and gate_class > max_val:
                    max_val = gate_class
                    
                    positive_roi = [(x, y, w, h)]
                    positive_image_roi = frame[y:y + h, x:x + w, :]

                    #positive_roi = [(x_sq, y_sq, w_sq, h_sq)]
                    #positive_image_roi = frame[y_sq:y_sq + h_sq, x_sq:x_sq + w_sq, :]
                    
                    cv2.imwrite(store_roi_to + str(roi_counter) + "_resized"  + ".jpg", roi_resized) # save roi to disk
                    #positive_roi.append( (x, y, w, h) )
                    predicted_counter += 1
                    print("\n#", predicted_counter, "Prediction %", gate_class, "\n")
                roi_counter += 1

            # OPTIONAL
            draw_rectangles(frame, positive_roi, threshold_color, 5, 5) # last 2 params are offset
            #draw_rectangles(frame, roi, threshold_color, 5, 5) # last 2 params are offset
            #draw_rectangles(frame, frame_filtered_boxes, all_cont_color, 5, 5) # last 2 params are offset

            '''
            # CENTER CIRCLE
            # draw circle in center of camera frame
            frame_row, frame_col, frame_ch = frame.shape
            cv2.circle(frame, (frame_col//2, frame_row//2), 50, (0, 0, 255), 2)
            circle_rad = 50
            area = np.pi * circle_rad
            
            # draw a circle in the center of the detected roi
            if (positive_roi != []):
                roi_x, roi_y, roi_w, roi_h = positive_roi[0] # coordinates of roi
                roi_row = (roi_x + (roi_x + roi_w) )//2 # center of x
                roi_col = (roi_y + (roi_y + roi_h) )//2 # center of y
                if (roi_row < 2):
                    asdf = 1
                else:
                    print()
                    cv2.circle(frame, (roi_row, roi_col), 5, (255, 0, 0), -1)
            # END CIRCLE
'''

            # write to file
            out.write(frame)
            
            ''' ------ VIEW MULTIPLE TEST SCREENS ------ '''
            cv2.imshow("gate", frame) # actual frame
            #cv2.resizeWindow("Gate", 100, 100)
            cv2.moveWindow("gate", 0, 0)
            
            cv2.imshow("thresholding", frame_thresh) # threshold frame
            #cv2.imshow("thresholding", frame_thresh_wat) # threshold frame
            cv2.moveWindow("thresholding", 700, 0)

            cv2.imshow("grayscale", video_frame_gray) # grayscale
            #cv2.imshow("grayscale", frame_blur) # grayscale
            cv2.moveWindow("grayscale", 0, 500)

            cv2.imshow("contours", frame_copy) # contours
            cv2.moveWindow("contours", 700, 500)
            ''' ------ /END VIEW MULTIPLE TEST SCREENS ------ '''

            if(cv2.waitKey(1) & 0xFF == ord("q") ):
                break
        else:
            break

    #print(svm.classes_)
    print("\nNumber of positive predictions:", predicted_counter, "\n")
    out.release()
    video.release()
    cv2.destroyAllWindows()
 
