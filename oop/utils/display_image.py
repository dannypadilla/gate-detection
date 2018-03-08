'''
Usage:
  display_image.py [<img-path>]

Output:
  an image is displayed in a SEPARATE screen
  * if no arg is provided, defaults to an image

  <ESC> - to exit screen
'''

import numpy as np
import cv2
import sys


def print_image(img):
    cv2.imshow("image", img)
    k = cv2.waitKey(0) & 0xFF
    if(k == 27):
        cv2.destroyAllWindows()


if __name__ == '__main__':

    print(__doc__)

    try:
        fn = sys.argv[1]
    except:
        fn = "../../tmp_files/third_run/front01.jpg"

    img = cv2.imread(fn)
    print_image(img)
