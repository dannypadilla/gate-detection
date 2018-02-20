import numpy as np
import cv2
import os
from subprocess import check_output

a = check_output(["ls", "."]).decode("utf8")

print(a)
