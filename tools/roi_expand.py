''' bug rn '''
def roi_to_square(roi_list):
    new_roi = []
    for x, y, w, h in roi_list:
        print("\nmethod", x, y, w, h)
        if(w > h):
            y_diff = (w - h)// 2
            print("y", y_diff)
            #square_roi = (x, y - y_diff, w, h + y_diff)
            square_roi = (x, y - y_diff, w, h + y_diff)
            print("square roi", square_roi)
        elif(h > w):
            x_diff = (h - w)// 2
            print("x", x_diff)
            #square_roi = (x - x_diff, y, w + x_diff, h)
            square_roi = (x - x_diff, y, w + x_diff, h)
            print("square roi", square_roi)
        else:
            print("Image is already square!")
            square_roi = (x, y, w, h)
        new_roi.append(square_roi)
    return new_roi


''' this works but if dim is odd... it's off by one.. not square '''
# maybe can resize after? to make square for odd dimensions
def roi_to_square_imgs(img, roi_list):
    new_imgs = []
    for x, y, w, h in roi_list:
        #print(x, y, w, h)
        if(w > h):
            y_diff = (w - h) // 2
            square_img = img[y - y_diff:y + h + y_diff, x:x + w, :]
        elif(h > w):
            x_diff = (h - w) // 2
            square_img = img[y:y + h, x - x_diff:x + w + x_diff, :]
        else:
            print("Image is already square!")
            square_img = img[y: y + h, x: x + w, :]
        new_imgs.append(square_img)
    return new_imgs
