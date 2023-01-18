import os
import cv2
import math
import matplotlib.pyplot as pl
import pandas as pd
from PIL import Image
import numpy as np
  
# Detect face
def face_detection(img):
    # faces = face_detector.detectMultiScale(img, 1.1, 4)
    # if (len(faces) <= 0):
    #     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     return img, img_gray
    # else:
    #     X, Y, W, H = faces[0]
    #     img = img[int(Y):int(Y+H), int(X):int(X+W)]
    #     return img, cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray
  
  
def trignometry_for_distance(a, b):
    return math.sqrt(((b[0] - a[0]) * (b[0] - a[0])) +\
                     ((b[1] - a[1]) * (b[1] - a[1])))
  
# Find eyes
def Face_Alignment(img):
    pl.imshow(img)
    pl.show()
    img_raw = img.copy()
    img, gray_img = face_detection(img)
    eyes = eye_detector.detectMultiScale(gray_img)
  
    # for multiple people in an image find the largest 
    # pair of eyes
    if len(eyes) >= 2:
        eye = eyes[:, 2]
        container1 = []
        for i in range(0, len(eye)):
            container = (eye[i], i)
            container1.append(container)
        df = pd.DataFrame(container1, columns=[
                          "length", "idx"]).sort_values(by=['length'])
        eyes = eyes[df.idx.values[0:2]]
  
        # deciding to choose left and right eye
        eye_1 = eyes[0]
        eye_2 = eyes[1]
        if eye_1[0] > eye_2[0]:
            left_eye = eye_2
            right_eye = eye_1
        else:
            left_eye = eye_1
            right_eye = eye_2
  
        # center of eyes
        # center of right eye
        right_eye_center = (
            int(right_eye[0] + (right_eye[2]/2)), 
          int(right_eye[1] + (right_eye[3]/2)))
        right_eye_x = right_eye_center[0]
        right_eye_y = right_eye_center[1]
        cv2.circle(img, right_eye_center, 2, (255, 0, 0), 3)
  
        # center of left eye
        left_eye_center = (
            int(left_eye[0] + (left_eye[2] / 2)), 
          int(left_eye[1] + (left_eye[3] / 2)))
        left_eye_x = left_eye_center[0]
        left_eye_y = left_eye_center[1]
        cv2.circle(img, left_eye_center, 2, (255, 0, 0), 3)
  
        # finding rotation direction
        if left_eye_y > right_eye_y:
            print("Rotate image to clock direction")
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1  # rotate image direction to clock
        else:
            print("Rotate to inverse clock direction")
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1  # rotate inverse direction of clock
  
        cv2.circle(img, point_3rd, 2, (255, 0, 0), 2)
        a = trignometry_for_distance(left_eye_center, 
                                     point_3rd)
        b = trignometry_for_distance(right_eye_center, 
                                     point_3rd)
        c = trignometry_for_distance(right_eye_center, 
                                     left_eye_center)
        cos_a = (b*b + c*c - a*a)/(2*b*c)
        angle = (np.arccos(cos_a) * 180) / math.pi
  
        if direction == -1:
            angle = 90 - angle
        else:
            angle = -(90-angle)
  
        # rotate image
        new_img = Image.fromarray(img_raw)
        new_img = np.array(new_img.rotate(direction * angle))
    else:
        new_img = img
  
    return new_img

from sklearn.datasets import fetch_lfw_people, fetch_olivetti_faces
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2

DATASET = "LFW" #Dataset ot use: LFW or OLIVETTI

####### Loading and parsing the dataset images #######
if DATASET == "LFW":
    lfw_people = fetch_lfw_people(color=True, min_faces_per_person=10, resize=0.5)
    X = lfw_people.images
    y = lfw_people.target
    X = np.array(X * 255, dtype='uint8')
elif DATASET == "OLIVETTI":
    lfw_people = fetch_olivetti_faces()
    X = lfw_people.images
    y = lfw_people.target
    X = np.array(X * 255, dtype='uint8')
    X = np.array([cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) for image in X])
else:
    raise ValueError(f"Dataset must be LFW or OLIVETTI, not {DATASET}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
  
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
nose_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_mcs_nose.xml")
  
for i in X_test:
    alignedFace = Face_Alignment(i)
    pl.imshow(alignedFace)
    pl.show()
    img, gray_img = face_detection(alignedFace)
    pl.imshow(img)
    pl.show()