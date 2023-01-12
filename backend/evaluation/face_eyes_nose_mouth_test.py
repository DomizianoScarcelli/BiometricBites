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

# get the faces detected in the image
def capture_face_features(img):
    face_hist = extract_histogram(img)
    eyes_hist = []
    nose_hist = []
    mouth_hist = []

    # eyes
    eyes = eyes_detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)
    if len(eyes) >= 2:
        x, y, w, h = eyes[0]
        eye_1 = extract_histogram(img[y:y+h, x:x+w])
        x, y, w, h = eyes[1]
        eye_2 = extract_histogram(img[y:y+h, x:x+w])

        eyes_hist = np.concatenate((eye_1, eye_2))
    else:
        x, y, w, h = eyes[0]
        eye_1 = extract_histogram(img[y:y+h, x:x+w])
        eye_2 = np.zeros(len(eye_1))
        eyes_hist = np.concatenate((eye_1, eye_2))

    # nose
    nose = nose_detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)
    if len(nose) != 0:
        x, y, w, h = nose[0]
        nose_hist = extract_histogram(img[y:y+h, x:x+w])
    else:
        nose_hist = np.zeros(len(nose))

    # mouth
    mouth = mouth_detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)
    if len(mouth) != 0:
        x, y, w, h = mouth[0]
        mouth_hist = extract_histogram(img[y:y+h, x:x+w])
    else:
        mouth_hist = np.zeros(len(mouth))
    print(len(face_hist), len(eyes_hist), len(nose_hist), len(mouth_hist))

    # concatenate hists
    conc_hist = np.concatenate([face_hist, eyes_hist, nose_hist, mouth_hist])

    return conc_hist

def extract_histogram(img):
    tmp_model = cv2.face.LBPHFaceRecognizer_create(
                    radius = 1, # The radius used for building the Circular Local Binary Pattern. The greater the radius, the smoother the image but more spatial information you can get
                    neighbors = 8, # The number of sample points to build a Circular Local Binary Pattern. An appropriate value is to use 8 sample points. Keep in mind: the more sample points you include, the higher the computational cost
                    grid_x = 8, # The number of cells in the horizontal direction, 8 is a common value used in publications. The more cells, the finer the grid, the higher the dimensionality of the resulting feature vector
                    grid_y = 8, # The number of cells in the vertical direction, 8 is a common value used in publications. The more cells, the finer the grid, the higher the dimensionality of the resulting feature vector
                )  
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

            img_i_hist = capture_face_features(img_i)
            img_j_hist = capture_face_features(img_j)

            # axis_values = np.array([i for i in range(0, len(img_i_hist))])
            # fig = plt.figure()
            # plt.bar(axis_values, img_i_hist)
            # plt.show()    
            
            print(len(img_i_hist), len(img_j_hist))        

            print(get_correlation_between_two(img_i_hist, img_j_hist))
            
            fig = plt.figure()
 
            ax1 = fig.add_subplot(1,2,1)
            plt.imshow(img_i)
            plt.axis('off')
            
            ax2 = fig.add_subplot(1,2,2)
            plt.imshow(img_j)
            plt.axis('off')
            
            plt.show()