import numpy as np
import cv2
from preprocess import Preprocess

class Contours(Preprocess.Preprocess):

    retrieval_mode_list = {
        1: "cv2.RETR_EXTERNAL", # gets only extreme outer contours
        2: "cv2.RETR_LIST", # gets all contours with no hierarchy
        3: "cv2.RETR_CCOMP",
        4: "cv2.RETR_TREE", # gets all contours and reconstructs full hierarchy <-- using this one for now
        5: "cv2.RETR_FLOODFILL" # ? don't use
    }
    approx_method_list = {
        1: "cv2.CHAIN_APPROX_NONE", # stores all the contour points
        2: "cv2.CHAIN_APPROX_SIMPLE" # compresses contours <-- using this one for now
    }

    def __init__(self, image_path, retrieval_mode, approx_method):
        super(Contours, self).__init__(image_path)
        self.image = super().get_image().copy() # trouble here... image sb grayscale/thresh image
        self.contours = None
        self.hierarchy = None
        self.retrieval_mode = retrieval_mode
        self.approx_method = approx_method
        self.color = None
        self.thickness = None
        self.filtered_contours = None

    def get_contours(self):
        return self.contours

    def preprocess(self):
        self.image, self.contours, self.hierarchy = cv2.findContours(
            super().get_grayscale_image(),
            self.retrieval_mode,
            self.approx_method
        )
        return (self.image, self.contours, self.hierarchy)

    # MAYBE overload to handle filtered_contours?
    #### DOESN'T WORK YET - image needs to be a threshold
    def draw_contours(self, color=(0, 255, 0), thickness=3):
        self.color = color
        self.thickness = thickness
        #image_copy = self.image.copy()
        cv2.drawContours(self.image, self.contours, -1, self.color, self.thickness)
        return self.image

    # returns a new list of contours filtered by boundary values (exclusive)
    def filter_contours(lower_boundary, upper_boundary):
        # need to error check if contours hasn't been calculated yet
        if (self.contours != None):
            self.filtered_contours = [] # empty existing list
            for cont in self.contours:
                if len(cont) > lower_boundary and len(cont) < upper_boundary:
                    self.filtered_contours.append(cont)
        return self.filtered_contours

    # finish
    def __str__(self): # not pointing to right ret-mode and approx-method
        return str(" * Contours:" +
                   "\n\tRetrieval Mode - " + str(self.retrieval_mode_list[self.retrieval_mode] ) +
                   "\n\tApproximation Method - " + str(self.approx_method_list[self.approx_method] ) +
                   "\n\tNumber of Contours - " + str((self.contours != None) ) +
                   "\n\tFiltered Contours - " + str((self.filtered_contours != None) )
        )
