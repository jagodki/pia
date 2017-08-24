#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 18:10:49 2017

@author: Christoph
"""

from PIL import Image
import face_recognition

# Load the jpg file into a numpy array
#image = face_recognition.load_image_file("/Users/Christoph/Desktop/Testbilder/iu.jpeg")
image = face_recognition.load_image_file("/Users/Christoph/Desktop/iu-4.jpeg")

#detect all faces
face_locations = face_recognition.face_locations(image)
print("I found {} face(s) in this photograph.".format(len(face_locations)))


for face_location in face_locations:

    # Print the location of each face in this image
    top, right, bottom, left = face_location
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    # You can access the actual face itself like this:
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()

print("Finished ^o^")