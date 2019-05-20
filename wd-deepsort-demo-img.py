#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Untuk back compability ke python 2
from __future__ import division, print_function, absolute_import

import warnings
warnings.filterwarnings("ignore")

import os
from timeit import time
import sys
import cv2
import numpy as np
from PIL import Image
from yolo_img import YOLO

# Untuk algoritma Deep Sort yang diambil
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

#
# Definisi untuk cek file image
#

def detect_img(yolo):
    try:
        image = Image.open("1.jpg")
    except:
        print('[-] Open Error! Try again!')
    else:
        r_image = yolo.detect_image(image)
        r_image.show()

    yolo.close_session()


# ---------------
# Start di sini
# ---------------
if __name__ == '__main__':
    detect_img(YOLO())
