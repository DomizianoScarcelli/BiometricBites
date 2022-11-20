import numpy as np
import cv2
import os

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
            # ...pick its Region of Intrest (from eyes to mouth) and save it
            roi_gray = gray[y:y+h, x:x+w] 
            img_item = path + "/" + str(count) + "_gray.png"
            cv2.imwrite(img_item, roi_gray)

            # Applying different filters to each captured frame
            roi_color = frame[y:y+h, x:x+w]  
            filters(roi_color, count)
            
            # TODO: test new methods
            filters_new(roi_color, count)

            # Show rectangle
            cv2.rectangle(roi_color,(x,y),(x+w,y+h),(255,0,0),2)

            count +=1

        # Display the resulting frame
        cv2.imshow('frame',frame)

        # Exit if..
        if cv2.waitKey(1) & 0xFF == 27: # ...'ESC' pressed
            break
        elif count >= 100: # ...100 templates saved
            break   

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
        
def filters(roi_color, count):
    # applying hue saturation
    hue_sat = apply_hue_saturation(roi_color.copy(), alpha=3, beta=3)
    img_item = path + "/" + str(count) + "_hue_sat.png"
    cv2.imwrite(img_item, hue_sat)

    # Applying sepia flter
    sepia = apply_sepia(roi_color.copy(), intensity=.8)
    img_item = path + "/" + str(count) + "_sepia.png"
    cv2.imwrite(img_item, sepia)

    # Apply color overlay
    color_overlay = apply_color_overlay(roi_color.copy(), intensity=.8, red=123, green=231)
    img_item = path + "/" + str(count) + "_color_overlay.png"
    cv2.imwrite(img_item, color_overlay)
    
    # Apply invert filter
    invert = apply_invert(roi_color.copy())
    img_item = path + "/" + str(count) + "_invert.png"
    cv2.imwrite(img_item, invert)

def filters_new(roi_color, count):
    return None

user = input("Digita nome e cognome: ")
user = user.replace(" ", "-").lower()

# image parameters
path = BASE_DIR + "/Capture/images/" + user
os.makedirs(path, exist_ok=True)

print("Inizio a catturare frame, premi ESC per uscire")
startCapturing()

print("Fine cattura")