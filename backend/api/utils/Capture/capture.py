import numpy as np
import cv2
import os
import tensorflow as tf
from numpy import asarray

BASE_DIR = os.path.dirname(os.path.dirname( __file__ ))

# Classifier
face_cascade = cv2.CascadeClassifier(BASE_DIR + '/cascades/data/haarcascade_frontalface_default.xml')

def capture(frame):
    # group frames
    frames = []
        
    image = tf.cast(tf.convert_to_tensor(frame), tf.uint8)
    image_gray = tf.image.rgb_to_grayscale(image)

    # image_crop = tf.image.crop_to_bounding_box(image_gray, 34, 0, 160, 160)
    frames.append(frame)
    frames.append(asarray(image_gray))

    # ...by applying different filters
    # Invert image
    # flip = horizontal_flip(image_gray)
    # frames.append(flip)

    # # Boosting constrast
    # # +0.9
    # contrast = increase_contrast(image_gray, 0.9)
    # frames.append(contrast)

    # # +1.5
    # contrast = increase_contrast(image_gray, 1.5)
    # frames.append(contrast)

    # # +2
    # contrast = increase_contrast(image_gray, 2)
    # frames.append(contrast)

    # # Boosting brightness
    # # -0.5
    # brightness = increase_brightness(image_gray, -0.5)
    # frames.append(brightness)

    # # -0.2
    # brightness = increase_brightness(image_gray, -0.2)
    # frames.append(brightness)


    # # +0.2
    # brightness = increase_brightness(image_gray, 0.2)
    # frames.append(brightness)
        
    return frames

def horizontal_flip(frame):
    flip = tf.image.flip_left_right(frame)
    encF = tf.image.encode_jpeg(flip)

    return encF

def increase_contrast(frame, factor):
    contrast = tf.image.adjust_contrast(frame, factor)
    encC = tf.image.encode_jpeg(contrast)

    return encC

def increase_brightness(frame, delta):
    brightness = tf.image.adjust_brightness(frame, delta)
    encB = tf.image.encode_jpeg(brightness, )
    
    return encB