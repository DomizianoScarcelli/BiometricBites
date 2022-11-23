import numpy as np
import cv2
import tensorflow as tf

def verify_alpha_channel(frame):
    try:
        frame.shape[3] # Looking for the alpha channel
    except IndexError:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    return frame

############
# ADVANCED #
############

def apply_hue_saturation(frame, alpha, beta):
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    s.fill(199)
    v.fill(255)
    hsv_image = cv2.merge([h, s, v])

    out = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    frame = verify_alpha_channel(frame)
    out = verify_alpha_channel(out)
    cv2.addWeighted(out, 0.25, frame, 1.0, .23, frame)
    return frame

def apply_color_overlay(frame, intensity=0.5, blue=0, green=0, red=0):
    frame = verify_alpha_channel(frame)
    frame_h, frame_w, frame_c = frame.shape
    sepia_bgra = (blue, green, red, 1)
    overlay = np.full((frame_h, frame_w, 4), sepia_bgra, dtype='uint8')
    cv2.addWeighted(overlay, intensity, frame, 1.0, 0, frame)
    return frame

def apply_sepia(frame, intensity=0.5):
    frame = verify_alpha_channel(frame)
    frame_h, frame_w, frame_c = frame.shape
    sepia_bgra = (20, 66, 112, 1)
    overlay = np.full((frame_h, frame_w, 4), sepia_bgra, dtype='uint8')
    cv2.addWeighted(overlay, intensity, frame, 1.0, 0, frame)
    return frame

def alpha_blend(frame_1, frame_2, mask):
    alpha = mask/255.0 
    blended = cv2.convertScaleAbs(frame_1*(1-alpha) + frame_2*(1-alpha))
    return blended

def apply_invert(frame):
    return cv2.bitwise_not(frame)

#########
# BASIC #
#########

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
    encB = tf.image.encode_jpeg(brightness)
    
    return encB