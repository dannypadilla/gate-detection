import numpy as np
import cv2
import os
import time

#imgs_path = "./no_gate/"
imgs_path = "/Users/dannypadilla/Desktop/images/" # where original imgs are
file_name = "/Users/dannypadilla/Desktop/images/resized/" # resized images will be saved here
scale = (240, 180) # original (744, 480)

ls = [] # list of files in directory

with os.scandir(imgs_path) as it: # from python3 docs
    for entry in it:
        if not entry.name.startswith(".") and entry.is_file:
            ls.append(entry.name)
ls_sorted = sorted(ls) # good enough for now

start_time = time.time()

for image_name in ls_sorted:
    current_path = imgs_path + image_name # this is the path where each image is
    img = cv2.imread(current_path)
    resized_img = cv2.resize(img, scale)
    cv2.imwrite(file_name + "resized_" + image_name, resized_img)
    print("Wrote " + file_name + image_name + " to " + current_path) # cmd line visuals

end_time = time.time()

elapsed = end_time - start_time

print("\nNumber of File(s) converted:", len(ls_sorted) )
print("Elapsed Time:", elapsed_time)

it.close()
cv2.destroyAllWindows() # just in case...
