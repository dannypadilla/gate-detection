import cv2
import numpy as np
import sys

def resize(img, scale): # resize image to scale value param
    return cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale) ))

def scaled(img, scale): # returns new scale value
    img_shape_x, img_shape_y, channels = img.shape
    if(img_shape_x > scale):
        return scale / img_shape_x
    else:
        return 1

def preprocess(image, lower_upper_list):
    lower = np.array(lower_upper_list[0], dtype="uint8")
    upper = np.array(lower_upper_list[1], dtype="uint8")

    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask = mask)

    return output, mask

def create_all_boxes(img_contours_list):
    box_list = []
    for contour in img_contours_list:
        cv2.boudingRect(contour)
        box_list.append(contour)
    return box_list

def filter_boxes(img_rectangle_list, filter_size=0):
    filtered_boxes = []
    for rectangle in img_rectangle_list:
        if(rectangle[2] * rectangle[3] > filter_size):
            filtered_boxes.append(rectangle)
    return filtered_boxes

def draw_rectangles(frame, frame_rectangle_list, x_offset = 0, y_offset = 0):
    for x, y, w, h in frame_rectangle_list:
        cv2.rectangle(
            frame,
            (x - x_offset, y - y_offset),
            ((x + x_offset) + w, (y + y_offset) + h),
            (0, 255, 0),
            2
        )

# **************************************************************** #

if __name__ == '__main__':

    print(__doc__)

    try:
        fn = sys.argv[1]
    except:
        fn = 0

    def nothing(*arg):
        pass
    
    video1_path = "img-to-video/run3.avi"
    video2_path = "img-to-video/run4.avi"

    video1 = cv2.VideoCapture(video2_path)
    lower_blue = np.array([55, 55, 55])
    upper_blue = np.array([150, 255, 255])
    threshold_color = [0, 255, 0] # green
    box_filter_size = 400

    while(video1.isOpened() ):
        ret, frame = video1.read()

        if(ret):
            # wat
            video_frame_1, mask = preprocess(frame, [lower_blue, upper_blue]) # preprocess
            video_frame_gray = cv2.cvtColor(video_frame_1, cv2.COLOR_BGR2GRAY) # gray
            ret, frame_thresh = cv2.threshold(video_frame_gray, 127, 255, cv2.THRESH_TOZERO)
            frame_c, frame_contours, frame_heirarchy = cv2.findContours(frame_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            #frame_copy = frame.copy()
            #cv2.drawContours(frame_copy, frame_contours, -1, threshold_color, 3)

            frame_all_boxes = [cv2.boundingRect(c) for c in frame_contours]
            frame_filtered_boxes = filter_boxes(frame_all_boxes, box_filter_size)
            draw_rectangles(frame, frame_filtered_boxes, 5, 5) # last 2 params are offset

            cv2.imshow("Run 3", frame)

            if(cv2.waitKey(1) & 0xFF == ord("q") ):
                break
        else:
            break

    video1.release()
    cv2.destroyAllWindows()
