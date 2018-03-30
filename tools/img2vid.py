import numpy as np
import cv2
import os

## need better numbering sorting [currently handles [01, 02, etc..] not [1, 2,.., 10, 11, etc..]
## handle cmd line args; path, file_name, fps... good enough for now tho
## on each frame add frame #, detection, telemetry?, etc.... 
## clean up plz

imgs_path = "./tmp_files/fourth_run"
file_name = "./created_files/run4_3fps.avi"
fps = 2.0 # has to be >= 1.0

front = "/" # for appending front_slash..... meh

#it = os.scandir(imgs_path)

ls = [] # list of files in directory

with os.scandir(imgs_path) as it: # from python3 docs
    for entry in it:
        if not entry.name.startswith(".") and entry.is_file:
            ls.append(entry.name)
ls_sorted = sorted(ls) # good enough for now

fourcc  = cv2.VideoWriter_fourcc(*"M", "J", "P", "G") # for mac
out = cv2.VideoWriter(file_name, fourcc, fps, (744, 480) ) # has to be frame size of img

for i in ls_sorted:
    new_path = imgs_path + front + i
    print("Wrote", new_path) # cmd line visuals
    img = cv2.imread(new_path)
    out.write(img)

it.close()
out.release()
cv2.destroyAllWindows() # just in case...
