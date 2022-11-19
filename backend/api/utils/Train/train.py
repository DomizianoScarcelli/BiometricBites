import os
import cv2
import pickle

from PIL import Image
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname( __file__ ))
image_dir = BASE_DIR + "/Capture/images"

# Classifier
# TODO: find better ones
face_cascade = cv2.CascadeClassifier(BASE_DIR + '/cascades/data/haarcascade_frontalface_default.xml')

# Recognizer
# TODO: find better ones
recognizer = cv2.face.LBPHFaceRecognizer_create() # Face recognizer model

current_id = 0
label_ids = {}
y_lables = [] # Number related to labels
x_train = [] # Numbers of the pixel values

print("Inizio training...")

# For each file in the image directory
for root, dirs, files in os.walk(image_dir):
    for file in files:
        path = os.path.join(root, file) # Save the path of each image
        label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower() # Save the label of each image
        
        # Assign id to labels
        if not label in label_ids: 
            label_ids[label] = current_id
            current_id += 1
        id_ = label_ids[label]

        # Turn the image into a numpy array
        image_array = np.array(Image.open(path), "uint8") 

        # TODO: tweak this
        faces = face_cascade.detectMultiScale(
            image_array, # Input grayscale image.
            scaleFactor=1.2, # Parameter specifying how much the image size is reduced at each image scale. It is used to create the scale pyramid.
            minNeighbors=5, # Parameter specifying how many neighbors each candidate rectangle should have, to retain it. A higher number gives lower false positives. 
            minSize=(20, 20) # Minimum rectangle size to be considered a face.
        )
        # Append the detected faces into x_train and their id into y_labels
        for (x, y, w, h) in faces:
            roi = image_array[y:y+h, x:x+w]
            x_train.append(roi)
            y_lables.append(id_)

# Save labels into file
with open(BASE_DIR + "/Train/pickles/face-labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

# Train items
recognizer.train(x_train, np.array(y_lables))
recognizer.save(BASE_DIR + "/Train/recognizers/face-trainner.yml")

print("Fine training")