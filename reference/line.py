import cv2
import numpy as np

img = np.zeros( (400, 400, 3), dtype = np.uint8)

cv2.line(img, (15, 15), (200, 200), (110, 220, 0), 2, 8 )

cv2.imshow("Image", img)

cv2.waitKey(0)


