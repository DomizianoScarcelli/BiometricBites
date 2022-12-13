import numpy as np
import cv2
import os
import time

BASE_DIR = os.path.dirname(os.path.dirname( __file__ ))

# Classifier
face_cascade = cv2.CascadeClassifier(BASE_DIR + '/cascades/data/haarcascade_frontalface_default.xml')

# image parameters
user = "danilo-corsi"
path = BASE_DIR + "/Capture/images/" + user
os.makedirs(path, exist_ok=True)

# video parameters
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_speed = 30 # higher number means faster frame rate
video_name = user + ".mp4"
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

writer = cv2.VideoWriter(video_name,fourcc, video_speed, (width,height))

def startCapturing(frame):
    count = 0

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    writer.write(frame)

# When everything done, release the capture
# writer.release()

def startCapturing(frame):
    # face recognition (from video)
    vid = cv2.VideoCapture(video_name)

    while vid.isOpened():
        # Capture frame-by-frame
        success, frame = vid.read()

        if success:
            # Convert picture into grayscale
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # TODO: tweak this
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

            # For each face...
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w] # ...pick its Region of Intrest (from eyes to mouth)

                # Save each face
                img_item = path + "/" + str(count) + ".png"
                cv2.imwrite(img_item, roi_gray)

                # TODO: apply different filters to each captured frame

                count +=1
        else: 
            break

# When everything done, release the video capture object
# vid.release()