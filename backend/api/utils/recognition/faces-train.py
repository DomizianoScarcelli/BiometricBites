import os
import cv2
import pickle

from PIL import Image
import numpy as np

# Classifier
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml') # frontal face
recognizer = cv2.face.LBPHFaceRecognizer_create() # face recognizer, work pretty well :)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

current_id = 0
label_ids = {}
y_lables = [] # number related to labels
x_train = [] # numbers of the pixel values

# For each file in the image directory
for root, dirs, files in os.walk(image_dir):
    for file in files:
        path = os.path.join(root, file) # save the path of each image
        label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower() # save the label of each image

        # assign id to labels
        if not label in label_ids: 
            label_ids[label] = current_id
            current_id += 1
        id_ = label_ids[label]

        pil_image = Image.open(path).convert("L") # turn image into grayscale
        
        # resize the images
        size = (550, 550)
        final_image = pil_image.resize(size, Image.Resampling.LANCZOS)

        image_array = np.array(final_image, "uint8") # turn the image into a numpy array

        # detect face into images
        faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
        for (x, y, w, h) in faces:
            roi = image_array[y:y+h, x:x+w]
            x_train.append(roi)
            y_lables.append(id_)

# save labels into file
with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

# train the item
recognizer.train(x_train, np.array(y_lables))
recognizer.save("trainner.yml")