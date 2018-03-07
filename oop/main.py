import numpy as np
import cv2
import sys
#import Gate, Task, Detector, Navigation, Preprocessor, Classifier, FeatureExtractor, Source
from preprocess import Preprocess as pp
from preprocess import Threshold as thresh
from preprocess import ColorFilter as cf

def print_image(img):
    cv2.imshow("image", img)
    k = cv2.waitKey(0) & 0xFF
    if(k == 27):
        cv2.destroyAllWindows()

def prepr_test(path):
    prep = pp.Preprocess(path)
    print(prep)

def thresh_test(path):
    threshold_pp = thresh.Threshold(path, 127, 255, cv2.THRESH_TOZERO)
    ret_val, img_thresh = threshold_pp.get_output()
    print_image(img_thresh)
    
if __name__ == '__main__':
    #cmd line args parsing

    path = "../tmp_files/third_run/front01.jpg"
    lower = [55, 55, 55]
    upper = [150, 255, 255]
    color_filter_pp = cf.ColorFilter(path, [55, 55, 55], [150, 255, 255] )
    im_filter, mask = color_filter_pp.get_output()
    print_image(im_filter)
    
    '''
    preprocess_list = [
        threshold_pp,
        color_filter
    ]
    source = Source()
    preprocessor = Preprocessor()
    classifier = Classifier()
    feature_extractor = FeatureExtractor()
    detector = Detector(source, preprocessor, classifier, feature_extractor)

    gate = Gate(detector)
    '''
