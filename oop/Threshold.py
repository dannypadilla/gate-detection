import numpy as np
import cv2

class Threshold(Preprocess):

    threshold_types = {
        0: 0,
        1: cv2.THRESH_BINARY,
        2: cv2.THRESH_BINARY_INV,
        3: cv2.THRESH_TRUNC,
        4: cv2.THRESH_TOZERO,
        5: cv2.THRESH_TO_ZERO_INV.
        6: cv2.THRESH_OTSU
    }

    def __init__(self, image, threshold_type, threshold_valu, max_value):
        Preprocess.__init__(self, image)
        self.threshold_type = threshold_type # fix how to get value from dict
        self.thresholf_value = threshold_value
        self.max_value = max_value
        self.ret_val, self.image_threshold = cv2.threshold(
            super().get_image_grayscale(),
            threshold_value,
            max_value,
            threshold_type
        )
    
    def get_threshold_type(self):
            return self.threshold_type
    
    # need to error check dict
    def set_threshold_type(self, threshold):
        self.threshold_type = self.threshold_types_list[threshold]
        super().set_image_grayscale()
        self.ret_val, self.image_threshold = cv2.threshold(
            super().get_image_grayscale(),
            self.threshold_value,
            self.max_value,
            self.threshold_type
        )

    def print_threshold_types_list(self):
        for num, thresh_type in self.threshold_types:
            print(num, thresh_type)

    def get_image_threshold():
        return self.image_threshold

    def get_return_value():
        return self.ret_value

    # finish
    def __str__(self):
        pass

    def __eq__(self, other):
        pass

    def __lt__(self, other):
        pass

    def __gt__(self, other):
        pass
