import numpy as np
import cv2
from classifier import Classifier

class SVM_Linear(Classifier.Classifier):

    def __init__(self): # params tbd
        super(SVM_Linear, self)__init__()
        self.kernel = "linear"

    def predict(self, sample):
        pass
