import pickle
import sys
import numpy as np
import face_recognition
import cv2
import os

BASE_DIR = os.path.dirname(os.path.dirname( __file__ ))

with open(BASE_DIR + '/Train/pickles/face-model.pickle', 'rb') as pickle_file:
    clf = pickle.load(pickle_file)

def recognize(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(img=rgb, model="hog")

    frame_encodings = face_recognition.face_encodings(face_image=rgb, known_face_locations=boxes)

    if frame_encodings:
        predictions = clf.predict_proba([frame_encodings[0]]).ravel()
        maxPred = np.argmax(predictions)
        confidence = predictions[maxPred]

        if confidence > 0.85:
            name = clf.predict([frame_encodings[0]])[0]
            print(name)
            print(confidence)
        else:
            name = "unknown"
            print(name)
            print(confidence)
            
    return frame