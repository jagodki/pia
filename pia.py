#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 18:10:49 2017

@author: Christoph
"""

from PIL import Image
import dlib
from skimage import io
from datetime import datetime


print("start processing: {}".format(datetime.now()))

cnn_face_detection_model = dlib.cnn_face_detection_model_v1("/Users/Christoph/Desktop/Testbilder/mmod_human_face_detector.dat")

img = io.imread("/Users/Christoph/Desktop/Testbilder/iu-2.jpeg")
dets = cnn_face_detection_model.cnn_face_detector(img, 1)

print("Number of faces detected: {}".format(len(dets)))

for i, d in enumerate(dets):
    left = d.left()
    right = d.right()
    top = d.top()
    bottom = d.bottom()

    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, left, top, right, bottom))

    face_image = img[top:bottom, left:right]
    pil_img = Image.fromarray(face_image)
    pil_img.show()

    print("time after processing picture: {}".format(datetime.now()))
    dlib.hit_enter_to_continue()

print("Finished ^o^")