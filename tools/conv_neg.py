'''
1. This program reads in a directory of images and converts each to grayscale

2. Outputs to directory

Usage:
  conv_neg.py <images/path/> <store/images/to/path>

Default:
  input:
    ../images/
  output:
    ../negative_crops/
'''
import cv2
import numpy as np
import pandas as pd
import sys
import os
import time


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
    import_imgs_time = time.time()
    img_num = 0
    img_list = []
    for img in img_names:
        print("\tTook", time.time() - import_imgs_time, "to import", img )
        img_list.append(cv2.imread(path + img) )
    return img_list


def to_square_img(img, color=255):
    x, y, ch = img.shape
    # handles vertical or horizontal
    if(x > y):
        x_new = (x - y) // 2 # calculates HALF of the missing margin size
        new_img = img[x_new:x - x_new, :y, :]
        
    elif(y > x):
        y_new = (y - x) // 2  # calculates HALF of the missing margin size
        new_img = img[:x, y_new:y - y_new, :]
    else:
        print("Image is already square!") # noob
        return img
    
    return new_img


## -- ## -- ## -- ## -- ## -- main -- ## -- ## -- ## -- ## -- ##

if __name__ == '__main__':
    start_time = time.time()

    print(__doc__)

    # too much error check?
    try:
        imgs_path = sys.argv[1]
        store_imgs_to = sys.argv[2]
    except:
        imgs_path = "../images/new/" # where files are located
        store_imgs_to = "../negative_crops/"

    try:
        imgs_file_names = get_file_names(imgs_path) # get list of files in directory
    except:
        print("Problem with path(s)")
        print(__doc__)

    get_files_names_time = time.time()
    print("\nGet files names elapsed time since start:", get_files_names_time - start_time, "\n")

    imgs_list = import_images(imgs_path, imgs_file_names) # convert each file in dir to image

    import_images_time = time.time()
    print("\nimport images elapsed time since start:", import_images_time - start_time, "\n")
    
    number_of_images = 0
    
    if(len(imgs_file_names) != len(imgs_list) ): # probably don't need..
        print("\nNumber of files in directory doesn't match number of images imported...")
        print("\nRIP\n")
        exit(0)
    else:
        number_of_images = len(imgs_list)

    # scale

    for img in range(number_of_images):
        scale = (80, 80)
        
        new_img = to_square_img(imgs_list[img]) # expand, combine
        
        gray = to_grayscale(new_img) # grayscale

        resized = cv2.resize(gray, scale) # resize
        #resized = cv2.resize(new_img, scale) # resize
        
        new_file_name = "new_" + imgs_file_names[img] # create new file name
        path_and_name = store_imgs_to + new_file_name
        
        cv2.imwrite(path_and_name, resized) # save
        
        print("Wrote: " + new_file_name + " to " +  store_imgs_to) # just cli visual

    end_time = time.time()

    elapsed = end_time - start_time

    print("\nTOOK:", elapsed, "to run...")
    print("\nDONE!\n")

    #cv2.imshow("new img", new_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
