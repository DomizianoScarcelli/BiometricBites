import numpy as np
import cv2 # "Hey! There is a face here"
import pickle

# Classifier
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml') # frontal face
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml') # eye 
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml') # smile 

recognizer = cv2.face_LBPHFaceRecognizer.create() # importing face recognizer
recognizer.read("trainner.yml")  # read the trained data

# Labels dictionary
labels = {"person_name": 1}
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f) 
    # inverting key with value
    labels = {v:k for k,v in og_labels.items()} 

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Turn captured frame into gray scale
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use the classifier to find faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    # Find the precise position of the faces
    for (x, y, w, h) in faces:
        # ROI: Region of interest, (ycord_start, ycord_end), the square in which the face was found
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = frame[y:y+h, x:x+w]

        # Use deep learned model to identify the person
        id_, conf = recognizer.predict(roi_gray)

        # if confidence is good...
        if conf >= 45 and conf <= 85:
            # ... write who we think we have recognized
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

        # Draw a rectangle around the face
        color = (255, 0, 0) #BGR 0-255 
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

        # eye detection
        eyes = eye_cascade.detectMultiScale(roi_gray)
        smile = smile_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in smile:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 2)

    # Display the camera frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'): # Close the frame by pressing 'q'
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()