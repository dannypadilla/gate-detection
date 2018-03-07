import numpy as np
import cv2
import time # date instead

class Preprocess():

    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(self.image_path)
        self.grayscale_image = None
        self.resize = None

    def get_image(self):
        return self.image

    def get_grayscale_image(self):
        if (self.grayscale_image == None):
            self.grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return self.grayscale_image

    def get_output(self):
        self.output = self.preprocess()
        return self.output

    def __str__(self):
        return str("Image Path: " + str(self.image_path) +
                   "\nBinary Image: " + str(self.grayscale_image != None)
        )
