from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
import time
import sys

from sklearn.svm import SVC

BASE_DIR = os.path.dirname(os.path.dirname( __file__ ))

def trainSVC(image_dir):
    knownEncodings = []
    knownNames = []

    for root, dirs, files in os.walk(image_dir):
        count = 0
        for file in files:
            print("[INFO] processing image {}/{}".format(count + 1, len(files)))
            path = os.path.join(root, file) # Save the path of each image
            name = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower() # Save the label of each image
            image = cv2.imread(path)

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(img=rgb, model="hog")

            frame_encodings = face_recognition.face_encodings(face_image=rgb, known_face_locations=boxes)

            if frame_encodings:
                knownEncodings.append(frame_encodings[0])
                knownNames.append(name)
            count += 1

    print("Stiamo generando il tuo file di encodings..")
    data = {"encodings": knownEncodings, "names": knownNames}

    f = open(BASE_DIR + "/Train/recognizers/face_encodings.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close()

    print("[INFO] start training face_encodings..")
    X = data['encodings']
    y = data['names']

    clf = SVC(C=1, kernel='linear', probability=True)

    clf.fit(X, y)

    print("Saving classifier to local folder")
    f = open(BASE_DIR + "/Train/pickles/face-model.pickle", "wb")
    f.write(pickle.dumps(clf))
    f.close()