import numpy as np
import cv2
import sys
#import Gate, Task, Detector, Navigation, Preprocessor, Classifier, FeatureExtractor, Source
from preprocess import Preprocess as pp
from preprocess import ColorFilter as cf
from preprocess import Threshold as thresh
from preprocess import Contours as cont
from preprocess import Preprocessor as ppsor
from detection import Detector as det

# ************************* METHODS ****************************** #

def print_image(img):
    cv2.imshow("image", img)
    k = cv2.waitKey(0) & 0xFF
    if(k == 27):
        cv2.destroyAllWindows()

def preprocessor_test(path):
    preprocessor = ppsor.Preprocessor()
    
    color_filter_pp = cf.ColorFilter(path, [55, 55, 55], [150, 255, 255] )
    threshold_pp = thresh.Threshold(path, 127, 255, cv2.THRESH_TOZERO)
    cont_pp = cont.Contours(path, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    preprocessor.add_preprocess(color_filter_pp)
    preprocessor.add_preprocess(threshold_pp)
    preprocessor.add_preprocess(cont_pp)
    
    for p in preprocessor.get_preprocessor_queue():
        print(p)
    

# *********************** END-METHODS **************************** #

if __name__ == '__main__':
    
    path = "../tmp_files/third_run/front01.jpg"
    preprocessor_test(path)

# ************************* TEST METHODS ****************************** #

def preprocess_test(path): # Preprocess()
    prep = pp.Preprocess(path)
    print(prep)

def color_filter_test(path): # ColorFilter()
    lower = [55, 55, 55]
    upper = [150, 255, 255]
    color_filter_pp = cf.ColorFilter(path, [55, 55, 55], [150, 255, 255] )
    im_filter, mask = color_filter_pp.get_output()
    print_image(im_filter)

def threshold_test(path): # Threshold()
    threshold_pp = thresh.Threshold(path, 127, 255, cv2.THRESH_TOZERO)
    ret_val, img_thresh = threshold_pp.get_output()
    print_image(img_thresh)

def contours_test(): # Contours()
    cont_pp = cont.Contours(path, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img, contours, ret = cont_pp.get_output()
    print(contours)
    print_image(img)
    # following doesn't work --- draw_contours doesn't work yet; image needs to be threshold
    #print_image(cont_pp.draw_contours() )
    #print_image(cont_pp.get_image() )
