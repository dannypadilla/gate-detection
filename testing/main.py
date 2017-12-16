"""
capture video
gray scale
binary

filter out color - MASK
* blue
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

def resize(img, scale):
    return cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale) ) )

def preprocess(image, low_up_list ):
    lower = np.array(low_up_list[0], dtype="uint8")
    upper = np.array(low_up_list[1], dtype="uint8")
    
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask = mask)
    
    return output, mask

image_test_1 = "../../image/fourth_run/front153.jpg" # gate and floor line below

image_test_nav1 = "../../image/fourth_run/front120.jpg" # navigation
image_test_nav2 = "../../image/fourth_run/front120.jpg" # nav
image_test_nav3 = "../../image/fourth_run/front120.jpg" # nav

image_test_detect1 = "../../img/third_run/front9.jpg" # detect
image_test_detect2 = "../../img/third_run/front10.jpg" # detect
image_test_detect3 = "../../img/third_run/front11.jpg" # detect curving
image_test_detect4 = "../../img/third_run/front12.jpg" # detect curve
image_test_detect5 = "../../img/third_run/front13.jpg" # detect curve

im = cv2.imread(image_test_detect2)
if im.shape[0] > 400:
    scale = 400.0 / im.shape[0]
else:
    scale = 1
    
im = resize(im, scale)
#plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB) )
#plt.show()

imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#plt.imshow(imgray, cmap='gray')
#plt.show()

flag, binaryImage = cv2.threshold(imgray, 85, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

th1 = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
th2 = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


edges = cv2.Canny(binaryImage, 50, 150)

titles = ["original", "grayscale", "edges", "binary img", "adap thresh MEAN_C", "adap thres GAUSSIAN_C"]
images = [im, imgray, edges, binaryImage, th1, th2]

# plt.imshow(edges, cmap='gray')
# plt.imshow(binaryImage, cmap='gray')
# #plt.imshow(th1, cmap='gray')
# plt.imshow(th2, cmap='gray')
# plt.show()


lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 (-b) )
    y1 = int(y0 + 1000 (a) )
    x2 = int(x0 - 1000 (-b) )
    y2 = int(x0 - 1000 (a) )

    cv2.line(im, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imwrite("hougelines3.jpg", im)

b = np.matrix(im[:,:,0]).sum()
g = np.matrix(im[:,:,1]).sum()
r = np.matrix(im[:,:,2]).sum()

red_ratio = float(r) / (b + g)
blue_ratio = float(b) / (r + g)
green_ratio = float(g) / (b + r)

red_val = int(red_ratio * 120)
green_val = int(green_ratio * 120)
blue_val = int(blue_ratio * 120)

#pimage, mask = preprocess(im, [ [0, 0, 0], [255, 255, red_val] ] ) # [b, g, r]
#pimage, mask = preprocess(im, [ [0, 0, 0], [100, green_val, 100] ] ) # [b, g, r]
#pimage, mask = preprocess(im, [ [0, 0, 0], [blue_val, 255, 255] ] ) # [b, g, r]

#plt.imshow(pimage)
#plt.show()

#imgray = cv2.cvtColor(pimage, cv2.COLOR_BGR2GRAY)
#plt.imshow(imgray, cmap='gray')
#plt.show()
