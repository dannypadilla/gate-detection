import numpy as np
import cv2

class Task():
    
    def __init__(self, detector):
        self.detector = Detector(
            detector.get_source(),
            detector.get_preprocessor(),
            detector.get_classifier,
            detector.get_feature_extractor
        )
        self.navigation = Navigation()
        self.is_complete = False

    def is_task_complete(self):
        return self.is_completed

    def get_detector(self):
        return self.detector

    def get_navigation(self):
        return self.get_navigation

    def __str__(self):
        pass


