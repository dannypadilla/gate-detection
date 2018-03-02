import numpy as np
import cv2
import os

imgs_path = "./tmp_files/third_run/"

file_name = "./resize/resized_"
scale = (186, 120) # original (744, 480)

ls = [] # list of files in directory

with os.scandir(imgs_path) as it: # from python3 docs
    for entry in it:
        if not entry.name.startswith(".") and entry.is_file:
            ls.append(entry.name)
ls_sorted = sorted(ls) # good enough for now

for image_name in ls_sorted:
    current_path = imgs_path + image_name # this is the path where each image is
    img = cv2.imread(current_path)
    
    resized_img = cv2.resize(img, scale)
    cv2.imwrite("resized_" + image_name, resized_img)
    print("Wrote " + file_name + image_name) # cmd line visuals

it.close()
cv2.destroyAllWindows() # just in case...
