'''
1. This program reads in a directory of images and converts each to grayscale

2. Outputs to directory

Usage:
  imgs_to_gray.py <images/path/> <store/images/to/path>

Default:
  input:
    ../tmp_files/
  output:
    ../images/converted/
'''
import cv2
import numpy as np
import pandas as pd
import sys
import os

## -- preprocess utils -- ##

def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## -- file utils -- ##

# input is a path to a directory of images
# takes a directory path and returns the file names in the dir as a list
def get_file_names(imgs_path, sort=False):
    # sort will come later..
    dir_list = []
    with os.scandir(imgs_path) as it:
        for entry in it:
            if not entry.name.startswith(".") and entry.is_file:
                dir_list.append(entry.name)
        return dir_list

# input is a path to images and a list of file names
# returns a list of images
def import_images(path, img_names):
    img_list = []
    for img in img_names:
        img_list.append(cv2.imread(path + img) )
    return img_list


## -- END file utils -- ##

if __name__ == '__main__':

    print(__doc__)

    try:
        imgs_path = sys.argv[1]
        store_imgs_to = sys.argv[2]
    except:
        imgs_path = "../tmp_files/" # where files are located
        store_imgs_to = "../images/converted/"

    try:
        imgs_file_names = get_file_names(imgs_path) # get list of files in directory
    except:
        print("Problem with path(s)")
        print(__doc__)

    imgs_list = import_images(imgs_path, imgs_file_names) # convert each file in dir to image

    number_of_images = 0
    
    if(len(imgs_file_names) != len(imgs_list) ):
        print("\nNumber of files in directory doesn't match number of images imported...")
        print("\nRIP\n")
        exit(0)
    else:
        number_of_images = len(imgs_list)
    
    for img in range(number_of_images):
        gray = to_grayscale(imgs_list[img] ) # convert to gray
        new_file_name = "gray_" + imgs_file_names[img]
        cv2.imwrite(store_imgs_to + new_file_name, gray)
        print("Wrote: " + new_file_name + " to " +  store_imgs_to)

    print("\nDONE!\n")
