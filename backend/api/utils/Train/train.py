import os
import cv2
import pickle

from PIL import Image
import numpy as np

# from bsproject import settings

# TODO: aggiusta percorsi assoluti
BASE_DIR = os.path.dirname(os.path.dirname( __file__ ))
image_dir = "/Users/dov/Library/Mobile Documents/com~apple~CloudDocs/dovsync/Documenti Universita/Biometric Systems/Project/Repos.nosync/BS-Project/backend/samples"

# Classifier
face_cascade = cv2.CascadeClassifier(BASE_DIR + '/cascades/data/haarcascade_frontalface_default.xml')

# Recognizer
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

        # Turn image into grayscale
        image = Image.open(path).convert("L")

        # Turn the image into a numpy array
        image_array = np.array(image, "uint8") 

        x_train.append(image_array)
        y_lables.append(id_)

        # Face recognition
        faces = face_cascade.detectMultiScale(
            image_array, # Input grayscale image.
            scaleFactor = 1.1, # Parameter specifying how much the image size is reduced at each image scale. It is used to create the scale pyramid.
            minNeighbors = 5, # Parameter specifying how many neighbors each candidate rectangle should have, to retain it. A higher number gives lower false positives. 
            minSize = (30, 30) # Minimum rectangle size to be considered a face.
        )

        # Append the detected faces into x_train and their id into y_labels
        for (x, y, w, h) in faces:
            roi = image_array[y:y+h, x:x+w]
            x_train.append(roi)
            y_lables.append(id_)

# Save labels into file
if os.path.exists(BASE_DIR + "/Train/pickles/face-labels.pickle"):
    os.remove(BASE_DIR + "/Train/pickles/face-labels.pickle")
with open(BASE_DIR + "/Train/pickles/face-labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

# Train items
TRAIN_DIR = BASE_DIR + "/Train/recognizers/face-trainner.yml"
# if os.path.exists(TRAIN_DIR):
#     recognizer.read(TRAIN_DIR)
#     recognizer.update(x_train, np.array(y_lables))
# else:
if os.path.exists(TRAIN_DIR):
    os.remove(TRAIN_DIR)
recognizer.train(x_train, np.array(y_lables))
recognizer.write(TRAIN_DIR)

print("Fine training")