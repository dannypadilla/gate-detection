import numpy as np
import cv2

class ColorFilter(Preprocess):
    
    def __init__(self, image_path, lower_bound, upper_bound):
        Preprocess.__init__(self, image_path)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.lower_bound_np = np.array(self.lower_bound, dtype="uint8")
        self.upper_bound_np = np.array(self.upper_bound, dtype="uint8")
        
        self.mask = cv2.inRange(
            super().get_image(),
            self.lower_bound_np,
            self.upper_bound_np
        )
        self.output = cv2.bitwise_and(
            super.get_image(),
            super.get_image(),
            mask=mask
        )

    def get_output(self):
        return self.output
    
    def get_bounds(self):
        return tuple(lower_bound, upper_bound)

    def get_mask(self):
        return self.mask

    def __str__(self):
        pass
