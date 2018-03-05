import numpy as np
import cv2

class Image():
    
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(self.image_path)
        self.image_shape = image.shape()

    def resize(self, scale):
        return cv2.resize(
            image,
            (int(image[1] * scale),
             int(image[0] * scale)
            )
            
        )

    
