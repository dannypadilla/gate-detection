import numpy as np
import cv2

path = "../../img/third_run/front9.jpg"

im = cv2.imread(path)
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0) # test more thresholds or use cv2.Canny()

# if cv2.CHAIN_APPROX_SIMPLE can find the contour lines of the gate then
# use cv2.CHAIN_APPROX_SIMPLE
# it's more costly, computation-wise, but has more accurate results
im2, contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(im, contours, 30, [0, 255, 0], 3)
cv2.drawContours(im, contours, 31, [0, 255, 0], 3)

# 0 - L
# 1 - R
cnt0 = contours[31] # L
cnt1 = contours[30] # R
M0 = cv2.moments(cnt0)
M1 = cv2.moments(cnt1)
centx0 = int(M0["m10"] / M0["m00"])
centy0 = int(M0["m01"] / M0["m00"])
centx1 = int(M1["m10"] / M1["m00"])
centy1 = int(M1["m01"] / M1["m00"])

for k, v in M0.items():
    print(k, v)
# print("Left centroid", centx0, ", ", centy0)
# print("Right centroid", centx1, ", ", centy1)

print("moments for 31 L:\n", M0)
print("\nmoments for 30 R:\n", M1)

x0, y0, w0, h0 = cv2.boundingRect(cnt0)
x1, y1, w1, h1 = cv2.boundingRect(cnt1)
# print(x0, y0, w0, h0)
# print(x1, y1, w1, h1)

bottommost = tuple(cnt1[cnt1[:, :, 1].argmax()][0] ) # extreme points
#print(bottommost)
cv2.rectangle(im, (x0, y0), (x1, centy1 + (centy1 - y1) ), (0, 0, 255), 3)
cv2.rectangle(im, (x0, y0), bottommost, (0, 0, 255), 3)

top_ten = []
print(" ######### ")
count = 0
for i in contours[30]:
    x = np.split(i, 1)
    #print(count, ": ", x)
    count += 1

cv2.imshow("im1", im)

k = cv2.waitKey(0) & 0xFF
if(k == 27):
    cv2.destroyAllWindows()
