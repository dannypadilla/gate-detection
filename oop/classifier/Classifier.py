import numpy as np
import cv2

class Classifier():
    
    def __init__(self): # params tbd
        self.data_set = None
        self.X_train = None
        self.X_testing = None
        self.y_train = None
        self.y_testing = None
        self.sample = None
        self.prediction = None
        self.model_path = None
        self.model = None

    def train():
        pass
    
    def get_output(self):
        return self.predict()

    def get_model(self):
        return self.model

    def save_model(self):
        pass

    def load_model(self):
        pass
