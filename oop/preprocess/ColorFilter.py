import numpy as np
import cv2
from preprocess import Preprocess

class ColorFilter(Preprocess.Preprocess):
    
    def __init__(self, image_path, lower_bound, upper_bound):
        super(ColorFilter, self).__init__(image_path)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.lower_bound_np = np.array(self.lower_bound, dtype="uint8")
        self.upper_bound_np = np.array(self.upper_bound, dtype="uint8")
        self.mask = None
        self.output = None
        

    def preprocess(self):
        self.mask = cv2.inRange(
            super().get_image(),
            self.lower_bound_np,
            self.upper_bound_np
        )
        self.output = cv2.bitwise_and(
            super().get_image(),
            super().get_image(),
            mask=self.mask
        )
        
        return (self.output, self.mask)
    
    def get_bounds(self):
        return tuple(lower_bound_value, upper_bound_value)

    def get_mask(self):
        return self.mask

    # finish
    def __str__(self):
        return str(
            "\n\tBoundary Values:"
        )
