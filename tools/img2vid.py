import numpy as np
import cv2
import os

## need better numbering sorting [currently handles [01, 02, etc..] not [1, 2,.., 10, 11, etc..]
## handle cmd line args; path, file_name, fps... good enough for now tho
## on each frame add frame #, detection, telemetry?, etc.... 
## clean up plz

imgs_path = "../no_gate/"
fps = 5.0 # has to be >= 1.0
file_name = "../created_files/no_gate_@" + str(fps) + "fps.avi"

#it = os.scandir(imgs_path)

ls = [] # list of files in directory

with os.scandir(imgs_path) as it: # from python3 docs
    for entry in it:
        if not entry.name.startswith(".") and entry.is_file:
            ls.append(entry.name)
ls_sorted = sorted(ls) # good enough for now

fourcc  = cv2.VideoWriter_fourcc(*"M", "J", "P", "G") # for mac
img_res = (744, 480)
out = cv2.VideoWriter(file_name, fourcc, fps, img_res) # has to be frame size of img

for i in ls_sorted:
    new_path = imgs_path + i
    img = cv2.imread(new_path)
    out.write(img)
    print("Wrote", new_path) # cmd line visuals

it.close()
out.release()
cv2.destroyAllWindows() # just in case...
