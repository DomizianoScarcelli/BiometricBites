import numpy as np
import os
import cv2
import pickle

print("Inizio riconoscimento...")

BASE_DIR = os.path.dirname(os.path.dirname( __file__ ))

# Classifier
# TODO: find better ones
face_cascade = cv2.CascadeClassifier(BASE_DIR + '/cascades/data/haarcascade_frontalface_default.xml')

# Recognizer
# TODO: find better ones
recognizer = cv2.face.LBPHFaceRecognizer_create() # face recognizer

# Read trained data
recognizer.read(BASE_DIR + "/Train/recognizers/face-trainner.yml")  

# Read labels dictionary
labels = {"person_name": 1}
with open(BASE_DIR + "/Train/pickles/face-labels.pickle", "rb") as f:
    og_labels = pickle.load(f) 
    labels = {v:k for k,v in og_labels.items()} # Inverting key with value

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

	# Turn captured frame into gray scale
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # TODO: tweak this
    faces = face_cascade.detectMultiScale(
        gray, # Input grayscale image.
        scaleFactor=1.2, # Parameter specifying how much the image size is reduced at each image scale. It is used to create the scale pyramid.
        minNeighbors=5, # Parameter specifying how many neighbors each candidate rectangle should have, to retain it. A higher number gives lower false positives. 
        minSize=(20, 20) # Minimum rectangle size to be considered a face.
    )

	# for each face...
    for (x, y, w, h) in faces:
		# ROI: Region of interest, (ycord_start, ycord_end), the square in which the face was found
    	roi_gray = gray[y:y+h, x:x+w] # ...pick its Region of Intrest (from eyes to mouth)

		# Use deep learned model to identify the person
    	id_, conf = recognizer.predict(roi_gray)

		# If confiden
		# ce is good...
    	if conf >= 45:
			# ... write who he think he recognized
    		font = cv2.FONT_HERSHEY_SIMPLEX
    		name = labels[id_]
    		color = (255, 255, 255)
    		stroke = 2
    		cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

		# Draw a rectangle around the face
    	color = (255, 0, 0) #BGR 0-255 
    	stroke = 2
    	end_cord_x = x + w
    	end_cord_y = y + h
    	cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    # Display the resulting frame
    cv2.imshow('frame',frame)

    # Exit if..
    if cv2.waitKey(1) & 0xFF == 27: # ...'ESC' pressed
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()