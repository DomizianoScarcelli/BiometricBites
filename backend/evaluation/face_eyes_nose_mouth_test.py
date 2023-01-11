from .evaluation import compute_similarities, open_set_identification_eval, verification_eval, verification_mul_eval
from sklearn.datasets import fetch_olivetti_faces
import numpy as np
from .plots import roc_auc_curve, far_frr_curve
from tqdm import tqdm
from scipy.spatial.distance import cosine
import os
import pandas as pd
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import cv2
from scipy.stats import pearsonr

eyes_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
nose_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_mcs_nose.xml")
mouth_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_mcs_mouth.xml")
print(cv2.data.haarcascades)

# get the faces detected in the image
def capture_face_features(img):
    face = img.copy()

    eyes_hist = []
    nose_hist = []
    mouth_hist = []
    f = plt.figure()

    # eyes
    eyes = eyes_detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)
    for (x,y,w,h) in eyes:
        plt.imshow(img[y:y+h, x:x+w])

        eyes_hist.append(extract_histogram(img[y:y+h, x:x+w]))
        # cv2.rectangle(face,(x,y),(x+w,y+h),(0,255,0),2)

        f.savefig("eye.png")


    # nose
    nose = nose_detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)
    if len(nose) != 0:
        x, y, w, h = nose[0]
        nose_hist = extract_histogram(img[y:y+h, x:x+w])
        # cv2.rectangle(face,(x,y),(x+w,y+h),(0,255,0),2)
        plt.imshow(img[y:y+h, x:x+w])

        f.savefig("nose.png")

    # mouth
    mouth = mouth_detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)
    if len(mouth) != 0:
        x, y, w, h = mouth[0]
        mouth_hist = extract_histogram(img[y:y+h, x:x+w])
        # cv2.rectangle(face,(x,y),(x+w,y+h),(0,255,0),2)
        plt.imshow(img[y:y+h, x:x+w])

        f.savefig("mouth.png")


def extract_histogram(img):
    tmp_model = cv2.face.LBPHFaceRecognizer_create(
                    radius = 1, # The radius used for building the Circular Local Binary Pattern. The greater the radius, the smoother the image but more spatial information you can get
                    neighbors = 8, # The number of sample points to build a Circular Local Binary Pattern. An appropriate value is to use 8 sample points. Keep in mind: the more sample points you include, the higher the computational cost
                    grid_x = 8, # The number of cells in the horizontal direction, 8 is a common value used in publications. The more cells, the finer the grid, the higher the dimensionality of the resulting feature vector
                    grid_y = 8, # The number of cells in the vertical direction, 8 is a common value used in publications. The more cells, the finer the grid, the higher the dimensionality of the resulting feature vector
                )  
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tmp_model.train([img], np.array([0]))
    histogram = tmp_model.getHistograms()[0][0]

    return histogram / max(histogram) # return normalized histogram

def get_correlation_between_two(hist1, hist2):
    return pearsonr(hist1, hist2)[0]

olivetti_people = fetch_olivetti_faces()
X = olivetti_people.images
y = olivetti_people.target
X = np.array(X*255, dtype='uint8')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

for i in range(0, len(X)):
    for j in range(0, len(X)):
        if y[i] == y[j]:    
            img_i = X[i]
            img_j = X[j]
            capture_face_features(img_i)
            capture_face_features(img_j)

            # img_i = detect_face(X[i])
            # img_j = detect_face(X[j])

            print(get_correlation_between_two(extract_histogram(img_i), extract_histogram(img_j)))
            
            fig = plt.figure()
 
            ax1 = fig.add_subplot(1,2,1)
            plt.imshow(img_i)
            plt.axis('off')
            
            ax2 = fig.add_subplot(1,2,2)
            plt.imshow(img_j)
            plt.axis('off')
            
            plt.show()