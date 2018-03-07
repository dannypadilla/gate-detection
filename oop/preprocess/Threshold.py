import numpy as np
import cv2
from preprocess import Preprocess as pp

class Threshold(pp.Preprocess):

    threshold_types_options = {
        0: 0,
        1: cv2.THRESH_BINARY,
        2: cv2.THRESH_BINARY_INV,
        3: cv2.THRESH_TRUNC,
        4: cv2.THRESH_TOZERO,
        5: cv2.THRESH_TOZERO_INV,
        6: cv2.THRESH_OTSU
    }

    def __init__(self, image_path, threshold_value, max_value, threshold_type):
        super(Threshold, self).__init__(image_path)
        self.threshold_value = threshold_value
        self.max_value = max_value
        self.threshold_type = threshold_type # fix how to get value from dict
        self.ret_val = None
        self.output = None

    # forcing grayscale might pose an issue?
    def preprocess(self):
        self.ret_val, self.output = cv2.threshold(
            super().get_grayscale_image(),
            self.threshold_value,
            self.max_value,
            self.threshold_type
        )
        return (self.ret_val, self.output)
    
    def get_threshold_type(self):
            return self.threshold_type
    
    # need to error check dict
    def set_threshold_type(self, threshold_type):
        self.threshold_type = self.threshold_type
        self.ret_val, self.output = cv2.threshold(
            super().get_grayscale_image(),
            self.threshold_value,
            self.max_value,
            self.threshold_type
        )

    def print_threshold_types_options(self):
        for num, thresh_type in self.threshold_types_options:
            print(num, thresh_type)

    def get_return_value():
        return self.ret_value

    # finish
    def __str__(self):
        return str(" * Threshold Params:" +
                   "\n\tType - " + str(self.threshold_type) +
                   "\n\tValue - " + str(self.threshold_value) +
                   "\n\tMax Value - " + str(self.max_value) +
                   "\n\tReturn Value - " + str(self.ret_val) +
                   "\n\tOutput - " + str(self.output != None)
        )

    def __eq__(self, other):
        pass
