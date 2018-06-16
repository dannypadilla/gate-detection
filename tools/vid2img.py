'''
1. Converts a video to many images
- need to add resize option

2. Outputs to a video file too

Usage:
  vid2img.py [<args_tbd>]

  Output is created in current directory
'''

import cv2
import numpy as np
import sys

def resize(frame, scale): # resize image to scale value param
    return cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale) ))

def scaled(frame, scale): # returns new scale value
    frame_shape_x, frame_shape_y, channels = frame.shape
    if(frame_shape_x > scale):
        return scale / frame_shape_x
    else:
        return 1


# **************************************************************** #

if __name__ == '__main__':

    print(__doc__)

    try:
        fn = sys.argv[1] # unused for now
    except:
        fn = 0

    ## these will eventually become cmdline args
    video_path = "../videos/auv_video/rawgate-07_output.avi"
    video = cv2.VideoCapture(video_path)

    ## for video output
    save_to_path = "../images/from_video/" # save images here
    file_name = "jon_gate_07" # base file name - unique chars are next
    frame_counter = 0 # for counting every frame in the video
    
    file_count = 1 # for file name counting - gets appended to stored file
    mod_frame = 30 # save every nth frame - if we don't want all frames of video saved

    while(video.isOpened() ):
        ret, frame = video.read()

        if(ret):
            #video_frame_gray = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY) # gray

            # only want every other 'mod_frame' frame
            if(frame_counter % mod_frame == 0):
                # path to write to
                whole_path = save_to_path + file_name + "_" + str(file_count) + ".jpg"
                # write to file
                cv2.imwrite(whole_path, frame)
                # cmd line visuals
                print("WROTE", whole_path, "TO", save_to_path)
                file_count += 1 # for naming

            frame_counter += 1 # for mod counting

            if(cv2.waitKey(1) & 0xFF == ord("q") ):
                break
        else:
            break

    video.release()
    cv2.destroyAllWindows()
