import numpy as np
import cv2
import os
import time

BASE_DIR = os.path.dirname(os.path.dirname( __file__ ))

# Classifier
# TODO: find better ones
face_cascade = cv2.CascadeClassifier(BASE_DIR + '/cascades/data/haarcascade_frontalface_default.xml')

# image parameters
user = input("Digita nome e cognome: ")
user = user.replace(" ", "-").lower()
path = "images/" + user
os.makedirs(path, exist_ok=True)
count = 0

# video parameters
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_speed = 30 # higher number means faster frame rate
video_name = user + ".mp4"
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

writer = cv2.VideoWriter(video_name,fourcc, video_speed, (width,height))

print("Inizio a catturare frame, premi ESC per terminare il riconoscimento")

while True:
    # Capture frame-by-frame
    ret,frame = cap.read()

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # TODO: tweak this
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    writer.write(frame)

    # For each face...
    for (x, y, w, h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == 27: # 27 = 'ESC'
        break

# When everything done, release the capture
cap.release()
writer.release()
cv2.destroyAllWindows()

time.sleep(3)

print("Elaboro il video...")

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
vid.release()

print("Elaborazione completata")