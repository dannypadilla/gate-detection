import numpy as np
import cv2
import os
import time

def to_square_img(img, color=255):

    x, y, ch = img.shape
    # handles vertical or horizontal
    if(x > y):
        y_new = (x - y) // 2 # calculates HALP of the missing margin size
        margin = np.zeros( (x, y_new2, 3), dtype="uint8" ) # array of margin size
        margin[:, :] = color # changes values to color

        add_top = np.hstack( (margin, img)) # add color margin to top of img
        add_bot = np.hstack( (add_top, margin) ) # add color margin to bottom of img
        
    elif(y > x):
        x_new = (y - x) // 2  # calculates HALF of the missing margin size
        margin = np.zeros( (x_new, y, 3), dtype="uint8") # array of margin size
        margin[:, :] = color # changes values to color
        
        add_top = np.vstack( (margin, img) ) # add color margin to the top of img
        add_bot = np.vstack( (add_top, margin) ) # add color margin to the bottom of the img
        
    else:
        print("Image is already square!") # noob
        return img
    
    return add_bot


### --- ### --- ### --- ### --- ### --- ### --- ###

''' expand, combine, grayscale, resize, save  '''

positive_img_path = "positive/adidas-both-00.jpg"

img = cv2.imread(positive_img_path)

# ops
new_img = to_square_img(img)

print("new image shape:", new_img.shape)

#cv2.imshow("new img", new_img)
cv2.imwrite("expanded.jpg", new_img)

#cv2.waitKey(0)
cv2.destroyAllWindows()
