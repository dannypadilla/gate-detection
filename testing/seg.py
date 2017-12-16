import cv2
import numpy as np
from matplotlib import pyplot as plt

def resize(img, scale):
    return cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale) ) )

x = "../../img/third_run/front9.jpg"

img = cv2.imread(x)

if img.shape[0] > 400:
    scale = 400.0 / img.shape[0]
else:
    scale = 1
    
img = resize(img, scale)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

cv2.imshow('frame', thresh)
