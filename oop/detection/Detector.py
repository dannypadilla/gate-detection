import numpy as np
import cv2

class Detector():

    def __init__(self, source, preprocessor, classifier, feature_extractor):
        self.source = Source()
        self.preprocessor = Preprocessor()
        self.classifier = Classifier()
        self.feature_extractor = FeatureExtractor()

    def get_source(self):
        return self.source

    def get_preprocessor(self):
        return self.preprocessor

    def get_classifier(self):
        return self.classifier

    def get_feature_extractor(self):
        return self.feature_extractor

