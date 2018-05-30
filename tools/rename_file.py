import os

print(__doc__)

try:
    imgs_path = sys.argv[1] # path to folder of images
except:
    imgs_path = "../no_gate/"

append_end = "@" + str(fps) + "fps.avi"

with os.scandir(imgs_path) as it: # from python3 docs
    for entry in it:
        if not entry.name.startswith(".") and entry.is_file:
            ls.append(entry.name)
            
ls_sorted = sorted(ls) # good enough for now

it.close()
