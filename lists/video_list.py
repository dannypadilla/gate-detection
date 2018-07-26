
'''
This is a helper file to declutter main
- The paths to each training sets are defined here
'''

''' VIDEO PATHS '''
video_dict = {
    1: "gate_jon_1.avi", # 0:10
    2: "gate_jon_2.avi", # 0:08
    3: "gate_jon_3.avi", # 1:31
    4: "no_gate_@5fps.avi", # 0:19
    5: "old_run4_@3fps.avi", # 0:56
    6: "gate-6.7.1_output.avi", # 6/7 test run
    7: "jon_gate_run_6.8_3.avi",
    8: "jon_gate_run_6.8_4.avi",
    9: "jon_gate_run_6.8_5.avi",
    10: "auv_video/rawgate-01_output.avi", # new lens 10 - 16
    11: "auv_video/rawgate-02_output.avi",
    12: "auv_video/rawgate-03_output.avi",
    13: "auv_video/rawgate-04_output.avi", # has gate
    14: "auv_video/rawgate-05_output.avi", # has gate
    15: "auv_video/rawgate-06_output.avi", # ripple test
    16: "auv_video/rawgate-07_output.avi", # has gate
    17: "auv_video/rawgate-08_output.avi", # school pool 6.18.18 V
    18: "auv_video/rawgate-09_output.avi",
    19: "auv_video/rawgate-10_output.avi",
    20: "auv_video/rawgate-11_output.avi",
    21: "auv_video/rawgate-12_output.avi",
    22: "auv_video/rawgate-13_output.avi",
    23: "auv_video/rawgate-14_output.avi",
    24: "auv_video/rawgate-15_output.avi",
    25: "auv_video/rawgate-16_output.avi",
    26: "auv_video/rawgate-17_output.avi",
    27: "auv_video/rawgate-18_output.avi",
    28: "auv_video/rawgate-19_output.avi",
    29: "auv_video/rawgate-20_output.avi", # school pool 6.18.18 ^
    30: "auv_video/rawgate-21_output.avi", # school pool 7.05.18
    31: "auv_video/rawgate-22_output.avi",
    32: "auv_video/rawgate-23_output.avi",
    33: "auv_video/rawgate-24_output.avi",
    34: "auv_video/rawgate-25_output.avi",
    35: "auv_video/rawgate-26_output.avi",
    36: "auv_video/rawgate-27_output.avi",
    37: "auv_video/rawgate-28_output.avi",
    38: "auv_video/rawgate-29_output.avi",
    39: "auv_video/rawgate-30_output.avi",
    40: "auv_video/rawgate-31_output.avi",
    41: "auv_video/rawgate-32_output.avi",
    42: "auv_video/rawgate-33_output.avi",
    43: "auv_video/rawgate-34_output.avi",
    44: "auv_video/rawgate-35_output.avi",
    45: "auv_video/rawgate-36_output.avi",
    46: "auv_video/rawgate-37_output.avi",
    47: "auv_video/rawgate-38_output.avi",
    48: "auv_video/rawgate-39_output.avi",
    49: "auv_video/rawgate-40_output.avi",
    50: "auv_video/rawgate-41_output.avi",
    51: "auv_video/rawgate-42_output.avi",
    52: "auv_video/rawgate-43_output.avi",
    53: "auv_video/rawgate-44_output.avi",
    54: "auv_video/rawgate-45_output.avi",
    55: "auv_video/rawgate-46_output.avi",
    56: "auv_video/rawgate-47_output.avi", # school pool 7.05.18 ^
    57: "auv_video/rawgate-48_output.avi", # school pool 7.16.18
    58: "auv_video/rawgate-49_output.avi",
    59: "auv_video/rawgate-50_output.avi",
    60: "auv_video/rawgate-51_output.avi",
    61: "auv_video/rawgate-52_output.avi",
    62: "auv_video/rawgate-53_output.avi",
    63: "auv_video/rawgate-54_output.avi",
    64: "auv_video/rawgate-55_output.avi",
    65: "auv_video/rawgate-56_output.avi",
    66: "auv_video/rawgate-57_output.avi",
    67: "auv_video/rawgate-58_output.avi",
    68: "auv_video/rawgate-59_output.avi",
    69: "auv_video/rawgate-60_output.avi",
    70: "auv_video/rawgate-61_output.avi",
    71: "auv_video/rawgate-62_output.avi",
    72: "auv_video/rawgate-63_output.avi",
    73: "auv_video/rawgate-64_output.avi", # school pool 7.16.18 ^
    74: "auv_video/rawdice-01.avi", # school pool 7.18.18
    75: "auv_video/rawdice-02.avi",
    76: "auv_video/rawdice-03.avi",
    77: "auv_video/rawdice-04.avi",
    78: "auv_video/rawdice-05.avi",
    79: "auv_video/rawdice-06.avi",
    80: "auv_video/rawdice-07.avi",
    81: "auv_video/rawdice-08.avi",
    82: "auv_video/rawdice-09.avi",
    83: "auv_video/rawdice-10.avi",
    84: "auv_video/rawdice-11.avi",
    85: "auv_video/rawdice-12.avi",
    86: "auv_video/rawdice-13.avi",
    87: "auv_video/rawdice-14.avi",
    88: "auv_video/rawdice-15.avi", # school pool 7.18.18 ^
    89: "auv_video/rawgate-65.avi", # 7.20.19  might be duplicate === V
    90: "auv_video/rawgate-66.avi",
    91: "auv_video/rawgate-67.avi",
    92: "auv_video/rawgate-68.avi",
    93: "auv_video/rawgate-69.avi",
    94: "auv_video/rawgate-70.avi",
    95: "auv_video/rawgate-71.avi",
    96: "auv_video/rawgate-72.avi",
    97: "auv_video/rawgate-73.avi,",
    98: "auv_video/rawgate-74.avi",
    99: "auv_video/rawgate-75.avi",
    100: "auv_video/rawgate-76.avi",
    101: "auv_video/rawgate-77.avi", # 7.20.19 ^
    102: "auv_video/rawgate-78.avi", # 7.23.18 v
    103: "auv_video/rawgate-79.avi",
    104: "auv_video/rawgate-80.avi",
    105: "auv_video/rawgate-81.avi",
    106: "auv_video/rawgate-82.avi",
    107: "auv_video/rawgate-83.avi",
    108: "auv_video/rawgate-84.avi",
    109: "auv_video/rawgate-85.avi",
    110: "auv_video/rawgate-86.avi",
    111: "auv_video/rawgate-87.avi",
    112: "auv_video/rawgate-88.avi",
    113: "auv_video/rawgate-89.avi",
    114: "auv_video/rawgate-90.avi",
    115: "auv_video/rawgate-91.avi",
    116: "auv_video/rawgate-92.avi",
    117: "auv_video/rawgate-93.avi",
    118: "auv_video/rawgate-94.avi",
    119: "auv_video/rawgate-95.avi",
    120: "auv_video/rawgate-96.avi",
    121: "auv_video/rawgate-97.avi",
    122: "auv_video/rawgate-98.avi",
    123: "auv_video/rawgate-99.avi",
    124: "auv_video/rawgate-100.avi",
    125: "auv_video/rawgate-101.avi",
    126: "auv_video/rawgate-102.avi",
    127: "auv_video/rawgate-103.avi",
    128: "auv_video/rawgate-104.avi", # 7.23.18 ^
    129: "auv_video/rawgate-105.avi", # 7.25.18 v
    130: "auv_video/rawgate-106.avi",
    131: "auv_video/rawgate-107.avi",
    132: "auv_video/rawgate-108.avi",
    133: "auv_video/rawgate-109.avi",
    134: "auv_video/rawgate-110.avi",
    135: "auv_video/rawgate-111.avi",
    136: "auv_video/rawgate-112.avi",
    137: "auv_video/rawgate-113.avi",
    138: "auv_video/rawgate-114.avi",
    139: "auv_video/rawgate-115.avi",
    140: "auv_video/rawgate-116.avi",
    141: "auv_video/rawgate-117.avi",
    142: "auv_video/rawgate-118.avi",
    143: "auv_video/rawgate-119.avi",
    144: "auv_video/rawgate-120.avi",
    145: "auv_video/rawgate-121.avi",
    146: "auv_video/rawgate-122.avi",
    147: "auv_video/rawgate-123.avi",
    148: "auv_video/rawgate-124.avi",
    149: "auv_video/rawgate-125.avi",
    150: "auv_video/rawgate-126.avi",
    151: "auv_video/rawgate-127.avi",
    152: "auv_video/rawgate-128.avi",
    153: "auv_video/rawgate-129.avi",
    154: "auv_video/rawgate-130.avi",
    155: "auv_video/rawgate-131.avi",
    156: "auv_video/rawgate-132.avi",
    157: "auv_video/rawgate-133.avi",
    158: "auv_video/rawgate-134.avi",
    159: "auv_video/rawgate-135.avi" # 7.25.18 ^
    

}
