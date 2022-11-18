import numpy as np
import cv2
import os

BASE_DIR = os.path.dirname(os.path.dirname( __file__ ))

# Classifier
# TODO: find better ones
face_cascade = cv2.CascadeClassifier(BASE_DIR + '/cascades/data/haarcascade_frontalface_default.xml')

user = input("Digita nome e cognome: ")
user = user.replace(" ", "-").lower()

# video parameters
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# image parameters
path = "images/" + user
os.makedirs(path, exist_ok=True)
count = 0

print("Inizio a catturare frame, premi ESC per uscire")

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

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

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        # TODO: apply different filters to each captured frame

        count +=1

    # Display the resulting frame
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == 27: # 27 = 'ESC'
        break

print("Fine cattura")

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()