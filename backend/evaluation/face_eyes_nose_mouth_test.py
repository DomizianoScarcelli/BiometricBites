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

def detect_eyes(img):
    eyes_hist = []

    eyes = eyes_detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)
    if len(eyes) >= 2:
        x, y, w, h = eyes[0]
        eye_1 = extract_histogram(img[y:y+h, x:x+w])
        x, y, w, h = eyes[1]
        eye_2 = extract_histogram(img[y:y+h, x:x+w])

        eyes_hist = np.concatenate((eye_1, eye_2))

    return eyes_hist
    
def detect_nose(img):
    nose_hist = []

    nose = nose_detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)
    if len(nose) != 0:
        x, y, w, h = nose[0]
        nose_hist = extract_histogram(img[y:y+h, x:x+w])

    return nose_hist

def detect_mouth(img):
    mouth_hist = []

    mouth = mouth_detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)
    if len(mouth) != 0:
        x, y, w, h = mouth[0]
        mouth_hist = extract_histogram(img[y:y+h, x:x+w])

    return mouth_hist

def capture_features(img_i, img_j):
    # count number of features detected
    img_i_eyes_hist = []
    img_i_nose_hist = []
    img_i_mouth_hist = []

    img_j_eyes_hist = []
    img_j_nose_hist = []
    img_j_mouth_hist = []

    # extract face histogram
    img_i_face_hist = extract_histogram(img_i)
    img_j_face_hist = extract_histogram(img_j)

    # detect eyes
    img_i_eyes_hist = detect_eyes(img_i)
    img_j_eyes_hist = detect_eyes(img_j)

    if (len(img_i_eyes_hist) != len(img_j_eyes_hist)):
        img_i_eyes_hist = []
        img_j_eyes_hist = []

    # detect nose
    img_i_nose_hist = detect_nose(img_i)
    img_j_nose_hist = detect_nose(img_j)

    if (len(img_i_nose_hist) != len(img_j_nose_hist)):
        img_i_nose_hist = []
        img_j_nose_hist = []

    #detect mouth
    img_i_mouth_hist = detect_mouth(img_i)
    img_j_mouth_hist = detect_mouth(img_j)

    if (len(img_i_mouth_hist) != len(img_j_mouth_hist)):
        img_i_mouth_hist = []
        img_j_mouth_hist = []
    
    # concatenate hists
    img_i_conc_hist = np.concatenate([img_i_face_hist, img_i_eyes_hist, img_i_nose_hist, img_i_mouth_hist])
    img_j_conc_hist = np.concatenate([img_j_face_hist, img_j_eyes_hist, img_j_nose_hist, img_j_mouth_hist])

    return img_i_conc_hist, img_j_conc_hist

def capture_features_with_score(img_i, img_j):
    # count number of features detected
    img_i_eyes_hist = []
    img_i_nose_hist = []
    img_i_mouth_hist = []

    img_j_eyes_hist = []
    img_j_nose_hist = []
    img_j_mouth_hist = []

    eyes_score = 0
    nose_score = 0
    mouth_score = 0

    # extract face histogram
    img_i_face_hist = extract_histogram(img_i)
    img_j_face_hist = extract_histogram(img_j)

    face_score = get_correlation_between_two(img_i_face_hist, img_j_face_hist)
    n_features = 1

    # detect eyes
    img_i_eyes_hist = detect_eyes(img_i)
    img_j_eyes_hist = detect_eyes(img_j)

    if (len(img_i_eyes_hist) != len(img_j_eyes_hist)):
        img_i_eyes_hist = []
        img_j_eyes_hist = []
    else:
        if(len(img_i_eyes_hist) != 0):
            eyes_score = get_correlation_between_two(img_i_eyes_hist, img_j_eyes_hist)
            n_features += 1

    # detect nose
    img_i_nose_hist = detect_nose(img_i)
    img_j_nose_hist = detect_nose(img_j)

    if (len(img_i_nose_hist) != len(img_j_nose_hist)):
        img_i_nose_hist = []
        img_j_nose_hist = []
    else:
        if(len(img_i_nose_hist) != 0):
            nose_score = get_correlation_between_two(img_i_nose_hist, img_j_nose_hist)
            n_features += 1

    #detect mouth
    img_i_mouth_hist = detect_mouth(img_i)
    img_j_mouth_hist = detect_mouth(img_j)

    if (len(img_i_mouth_hist) != len(img_j_mouth_hist)):
        img_i_mouth_hist = []
        img_j_mouth_hist = []
    else:
        if(len(img_i_mouth_hist) != 0):
            mouth_score = get_correlation_between_two(img_i_mouth_hist, img_j_mouth_hist)
            n_features += 1

    overall_score = (face_score + eyes_score + nose_score + mouth_score) / n_features
    
    return overall_score

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

# def get_correlation_between_two(hist1, hist2):
#     min_sum = sum(np.minimum(hist1, hist2))
#     return 0.5 * min_sum * ((1/sum(hist1)) + (1/sum(hist2))) # computing intersection between hist1 and hist2

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

            # img_i_hist = extract_histogram(img_i)
            # img_j_hist = extract_histogram(img_j)
            # np.savetxt('img_i_hist.txt', img_i_hist)
            # np.savetxt('img_j_hist.txt', img_j_hist)

            img_i_hist, img_j_hist = capture_features(img_i, img_j)
            print(get_correlation_between_two(img_i_hist, img_j_hist))

            # print(capture_features_with_score(img_i, img_j))
            
            fig = plt.figure()
 
            ax1 = fig.add_subplot(1,2,1)
            plt.imshow(img_i)
            plt.axis('off')
            
            ax2 = fig.add_subplot(1,2,2)
            plt.imshow(img_j)
            plt.axis('off')
            
            plt.show()