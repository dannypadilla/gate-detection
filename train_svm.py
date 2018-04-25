'''
1. This program will train an SVM on training data passed in and store the model to disk

2, Training data (images) are expected to be .jpg files

* Usage:
  train_svm.py <positive/images/path/> <negative/images/path/> <model_name>

* Defaults:
    - Positive Images:
        ./images/gate/positive/

    - Negative Images:
        ./images/gate/negative/

    - Model Path:
        ./models/gate/

* Output:
   models/gate/
'''

import cv2
import numpy as np
import pandas as pd
import sys
import glob
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# **************************** USEFUL METHODS ********************* #


def resize(frame, scale):
    return cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale) ))


def get_features_with_label(img_data, hog, label):
    dims = (80, 80)
    data = []
    for img in img_data:
        img = cv2.resize(img, dims)
        feat = hog.compute( img[:, :, :] )
        data.append( (feat, label) )
    return data


def init_hog():
    min_dims = 80
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    bins = 9
    dims = (80, 80)
    hog = cv2.HOGDescriptor(dims, block_size, block_stride, cell_size, bins)
    return hog


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

def model_tests():
    pass
    

# ****************************** /END METHODS/ ******************** #

# ***************************************************************** #

if __name__ == '__main__':

    print(__doc__)

    # see if args were passed
    try:
        positive_images_path = sys.argv[1]
        negative_images_path = sys.argv[2]
        model_name = sys.argv[3]
    except:
        # defaults for my machine
        positive_images_path = "jupyter/positive/*.jpg"
        negative_images_path = "jupyter/negative/*.jpg"
        model_name = "orig_svm"
    
    # init HOG - feature extractor
    hog = init_hog()

    # init SVM
    svm = None # SVM - defined for scope? maybe don't need this...?
    model_path = "models/gate/" # where models are stored (should be)
    model_file_name = model_name + ".pkl" # append file extension to the model_file name

    py_vers_label = "py2" # python 2 is default version
    # check python version
    if(sys.version_info >= (3, 0) ): # since joblib/pickle is picky with python versions
        py_vers_label = "py3" # and since I'm using python3 for testing on my mac

    path = model_path + py_vers_label + "_" + model_file_name # the entire path to the model appended (naming convention)
    
    # see if MODEL exists... if not TRAIN and STORE to disk
    print("\n MESSAGE(S):")
    try:
        svm = joblib.load(path)
        print("Model already exists!")
        print("\nExiting...\n")
    except:
        print("Training model...")
        svm = SVC(C=1.0, kernel="linear", probability=True, random_state=2)
        train_svm(svm, hog, positive_images_path, negative_images_path)
        joblib.dump(svm, path) # store model object to disk
        print("\n\tStored model to location: " + "\"" + path + "\"\n")

    
    # ******************************* /END OF CURRENT IMPLEMENTATION/ ************************ #

    
    '''
    # ****************************************************************** #
    # ### ### ### ### IN-WORK: DON'T USE YET ### ### ### ### ### ### ### #
    # * Next part will record from camera source and store to disk       #
    # * Need to determine whether we want to train on preprocessing data #
    # * --- Will need to change above model training to new source       #
    # * --- ALSO determine scale of frame/image source                   #
    # ****************************************************************** #

    # will be implemented as args
    # test for type of params passed... if bool.. etc..
    train_with_video_source = False
    camera_is_upside_down = False

    #video setup - NOT CURRENTLY IMPLEMENTED... YET
    # this will be the camera source - maybe using video would be too much at this point
    video_path = "videos/gate_new.avi" # not used right now
    video = cv2.VideoCapture(video_path) # not used right now
    
    ## for outputting video
    fps = 30.0
    file_name = "./run_jons_.avi"
    
    fourcc  = cv2.VideoWriter_fourcc(*"M", "J", "P", "G") # create write object for mac
    #out = cv2.VideoWriter(file_name, fourcc, fps, (744, 480) ) # has to be frame size of img
    out = cv2.VideoWriter(file_name, fourcc, fps, (640, 480) ) # has to be frame size of img

    

    while( (video.isOpened() ) and train_with_video_source):
        ret, frame = video.read()
        
        if(camera_is_upside_down): # whether camera should be rotated 180 deg
            rows, cols,_ = frame.shape
            rot_trans = cv2.getRotationMatrix2D( (cols/2, rows/2), 180, 1) # rotate image 180
            frame = cv2.warpAffine(frame, rot_trans, (cols, rows) ) # since camera is upside down..

        if(ret):
            #video_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # gray

            # loop here for processing each frame to SVM

            cv2.imshow("gate", frame) # actual frame
            cv2.moveWindow("gate", 0, 0)

            if(cv2.waitKey(1) & 0xFF == ord("q") ):
                break
        else:
            break

    #print(svm.classes_)
    out.release()
    video.release()
    cv2.destroyAllWindows()
    '''
