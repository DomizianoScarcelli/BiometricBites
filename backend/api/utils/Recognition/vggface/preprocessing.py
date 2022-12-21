import cv2
import os
import numpy as np
from PIL import Image

from bsproject.settings import SAMPLES_ROOT

image_width = 224
image_height = 224

# for detecting faces
facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

current_id = 0
label_ids = {}

# iterates through all the files in each subdirectories
for root, _, files in os.walk(SAMPLES_ROOT):
    for file in files:
        # path of the image
        path = os.path.join(root, file)

        # get the label name (name of the person)
        label = os.path.basename(root).replace(" ", ".").lower()

        # add the label (key) and its number (value)
        if not label in label_ids:
            label_ids[label] = current_id
            current_id += 1

        # load the image
        imgtest = cv2.imread(path, cv2.IMREAD_COLOR)
        print(imgtest)
        image_array = np.array(imgtest, "uint8")

        # get the faces detected in the image
        faces = facecascade.detectMultiScale(imgtest, scaleFactor=1.1, minNeighbors=5)

        # if not exactly 1 face is detected, skip this photo
        if len(faces) != 1: 
            print(f'---Photo skipped---\n')
            # remove the original image
            os.remove(path)
            continue

        # save the detected face(s) and associate
        # them with the label
        for (x_, y_, w, h) in faces:
            # resize the detected face to 224x224
            size = (image_width, image_height)

            # detected face region
            roi = image_array[y_: y_ + h, x_: x_ + w]

            # resize the detected head to target size
            resized_image = cv2.resize(roi, size)
            image_array = np.array(resized_image, "uint8")

            # remove the original image
            os.remove(path)

            # replace the image with only the face
            im = Image.fromarray(image_array)
            im.save(path)