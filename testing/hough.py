import cv2
import numpy as np

img = cv2.imread("", 0)
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
#cv2.imshow("image", img)

#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img, 100, 200, apertureSize = 3)
cv2.imshow("ed", edges)
lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b) )
    y1 = int(y0 + 1000 * (a) )
    x2 = int(x0 - 1000 * (-b) )
    y2 = int(y0 - 1000 * (a) )

    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imwrite("houghlines.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
