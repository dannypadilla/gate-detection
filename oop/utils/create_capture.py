#!/usr/bin/env python

'''
'create_capture' is a convinience function for capture creation,
falling back to procedural video in case of error.

Usage:
    video.py [--shotdir <shot path>] [source0] [source1] ...'

    sourceN is an
     - integer number for camera capture
     - name of video file

Synth examples:
    synth:bg=../data/lena.jpg:noise=0.1
    synth:class=chess:bg=../data/lena.jpg:noise=0.1:size=640x480

Keys:
    ESC    - exit
    SPACE  - save current frame to <shot path> directory

'''


from __future__ import print_function
import numpy as np
import cv2

def create_capture(source=0):
    ''' Don't think will need this... but w/e
    source: <int> or '<int>|<filename>|synth [:<param_name>=<value> [:...]]'
    '''
    source = str(source).strip() # removes leading and trailing whitespace
    chunks = source.split(':')
    # handle drive letter ('c:', ...)
    if (len(chunks) > 1 and len(chunks[0]) == 1 and chunks[0].isalpha() ):
        chunks[1] = chunks[0] + ':' + chunks[1]
        del chunks[0]

    source = chunks[0]
    try:
        source = int(source)
    except ValueError:
        pass
    params = dict( s.split('=') for s in chunks[1:] )

    cap = cv2.VideoCapture(source)
    if ('size' in params):
        w, h = map(int, params['size'].split('x'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if (cap is None or not cap.isOpened() ):
        print('Warning: unable to open video source: ', source)
        if fallback is not None:
            return create_capture(fallback, None)
    return cap
