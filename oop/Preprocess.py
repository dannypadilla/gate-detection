import numpy as np
import cv2
import time # date instead

class Preprocess():

    def __init__(self, image_path):
        # check if it's a path or nparray
        self.image = cv2.imread(image_path) 
        self.binary_image = None
        self.resize = None

    def get_image(self):
        return self.image

    def get_image_grayscale(self):
        if (self.binary_image == None):
            self.binary_image = cv2.toBG() # replace with actual code
        return self.binary_image

    def __str__(self):
        return (
            "Image Path:", str(self.image_path),
            "Binary Image?:", str(self.binary_image != None)
        )
