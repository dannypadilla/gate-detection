import numpy as np
import cv2
import os
import time

imgs_path = "./no_gate/"
file_name = "/negative_crops/"
scale = (744, 480)
window_size = (80, 80)

ls = [] # list of files in directory

with os.scandir(imgs_path) as it: # from python3 docs
    for entry in it:
        if not entry.name.startswith(".") and entry.is_file:
            ls.append(entry.name)

start_time = time.time()

for image_name in ls:
    for row_col in range(12):
    current_path = imgs_path + image_name # this is the path where each image is
    img = cv2.imread(current_path)
    resized_img = cv2.resize(img, scale)
    cv2.imwrite(file_name + "resized_" + image_name, resized_img)
    print("Wrote " + file_name + image_name + " to " + current_path) # cmd line visuals

end_time = time.time()

elapsed = end_time - start_time

print("\nNumber of File(s) converted:", len(ls) )
print("Elapsed Time:", elapsed_time)

it.close()
cv2.destroyAllWindows() # just in case...
