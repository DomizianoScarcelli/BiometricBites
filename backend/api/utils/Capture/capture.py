import numpy as np
import cv2
import os
import tensorflow as tf

from filters import *

BASE_DIR = os.path.dirname(os.path.dirname( __file__ ))

# Classifier
face_cascade = cv2.CascadeClassifier(BASE_DIR + '/cascades/data/haarcascade_frontalface_default.xml')

def startCapturing():
    # Video parameters
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 640) # Set video width
    cap.set(4, 480) # Set video height

    count = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert picture into grayscale
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face recognition
        faces = face_cascade.detectMultiScale(
            gray, # Input grayscale image.
            scaleFactor = 1.1, # Parameter specifying how much the image size is reduced at each image scale. It is used to create the scale pyramid.
            minNeighbors = 5, # Parameter specifying how many neighbors each candidate rectangle should have, to retain it. A higher number gives lower false positives. 
            minSize=(30, 30) # Minimum rectangle size to be considered a face.
        )

        # For each face...
        for (x, y, w, h) in faces:
            # ...pick its Region of Intrest (from eyes to mouth) and save it as...
            
            # ...gray image...
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]  

            img_item_gray = path + "/" + str(count) + "_gray.png"
            cv2.imwrite(img_item_gray, roi_gray)

            img_item_color = path + "/" + str(count) + "_color.png"
            cv2.imwrite(img_item_color, roi_color)

            image_gray = tf.io.read_file(img_item_gray)
            image_gray = tf.image.decode_png(image_gray, 3)
            # image_color = tf.io.read_file(img_item_color)
            # image_color = tf.image.decode_png(image_color, 3)

            # ...by applying different filters
            filters_basic(image_gray, count)
            # filters_basics(image_color, count)
            # filters_advanced(roi_color, count)

            # Show rectangle
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

            count +=1

        # Display the resulting frame
        cv2.imshow('frame',frame)

        # Exit if...
        if cv2.waitKey(1) & 0xFF == 27: # ...'ESC' pressed
            break
        elif count >= 20: # ...100 templates saved
            break   

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
        
def filters_advanced(image, count):
    # Applying hue saturation
    hue_sat = apply_hue_saturation(image.copy(), alpha=3, beta=3)
    img_item = path + "/" + str(count) + "_hue_sat.png"
    cv2.imwrite(img_item, hue_sat)

    # Applying sepia flter
    sepia = apply_sepia(image.copy(), intensity=.8)
    img_item = path + "/" + str(count) + "_sepia.png"
    cv2.imwrite(img_item, sepia)

    # Applying color overlay
    color_overlay = apply_color_overlay(image.copy(), intensity=.8, red=123, green=231)
    img_item = path + "/" + str(count) + "_color_overlay.png"
    cv2.imwrite(img_item, color_overlay)
    
    # Applying invert filter
    invert = apply_invert(image.copy())
    img_item = path + "/" + str(count) + "_invert.png"
    cv2.imwrite(img_item, invert)

def filters_basic(image, count):
    # Invert image
    flip = horizontal_flip(image)
    fnameF = tf.constant(path + "/" + str(count) + "_flip.png")
    tf.io.write_file(fnameF, flip)

    # Boosting constrast
    # +0.9
    contrast = increase_contrast(image, 0.9)
    fnameF = tf.constant(path + "/" + str(count) + "_cont1.png")
    tf.io.write_file(fnameF, contrast)

    # +1.5
    contrast = increase_contrast(image, 1.5)
    fnameF = tf.constant(path + "/" + str(count) + "_cont2.png")
    tf.io.write_file(fnameF, contrast)

    # +2
    contrast = increase_contrast(image, 2)
    fnameF = tf.constant(path + "/" + str(count) + "_cont3.png")
    tf.io.write_file(fnameF, contrast)

    # Boosting brightness
    # -0.5
    brightness = increase_brightness(image, -0.5)
    fnameF = tf.constant(path + "/" + str(count) + "_bright1.png")
    tf.io.write_file(fnameF, brightness)

    # -0.2
    brightness = increase_brightness(image, -0.2)
    fnameF = tf.constant(path + "/" + str(count) + "_bright2.png")
    tf.io.write_file(fnameF, brightness)

    # +0.2
    brightness = increase_brightness(image, 0.2)
    fnameF = tf.constant(path + "/" + str(count) + "_bright3.png")
    tf.io.write_file(fnameF, brightness)

print("Inizio cattura")

user = input("Digita nome e cognome: ")
user = user.replace(" ", "-").lower()

# image parameters
path = BASE_DIR + "/Capture/images/" + user
os.makedirs(path, exist_ok=True)

print("Inizio a catturare frame, premi ESC se vuoi uscire prima")
startCapturing()

print("Fine cattura")