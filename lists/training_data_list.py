
'''
This is a helper file to declutter main
- The paths to each training sets are defined here
'''


''' POSITIVE TRAINING DATASET PATHS '''
pos_img_dict = {
    1: "images/whole_gate/*.jpg",
    2: "images/bars/*.jpg",
    3: "images/whole_gate_and_bars/*.jpg",
    4: "images/gray_whole_gate/*.jpg",
    5: "images/gray_bars/*.jpg",
    6: "images/gray_whole_gate_and_bars/*.jpg",
    7: "jupyter/positive/*.jpg", # no jons pool data
    8: "jupyter/positive_old/*.jpg", # before resize to 80x80
    9: "images/stairs_pos_orig/*.jpg", # new lens - not resized
    10: "images/stairs_pos/*.jpg", # resized to 80x80
    11: "images/gray_stairs_pos/*.jpg",
    12: "images/all_positive/*.jpg", # all color images combined
    13: "images/gray_all_positive/*.jpg", # all gray images combined
    14: "images/red_gate_positive/*.jpg",
    15: "images/dice_all/*.jpg",
    16: "images/new_all_positive/*.jpg"
}


''' NEGATIVE TRAINING DATASET PATHS '''
neg_img_dict = {
    1: "images/negatives/*.jpg",
    2: "images/large_negatives/*.jpg",
    3: "images/gray_negatives/*.jpg",
    4: "images/stairs_neg_orig/*.jpg", # new lens - not resized
    5: "images/stairs_neg/*.jpg", # resized to 80x80
    6: "images/gray_stairs_neg/*.jpg",
    7: "images/all_negative/*.jpg", # all color images combined
    8: "images/gray_all_negative/*.jpg", # all gray images combined
    9: "images/red_gate_negative/*.jpg",
    10: "images/new_all_negatives/*.jpg"
}
