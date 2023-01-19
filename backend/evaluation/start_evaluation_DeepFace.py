from .evaluation import compute_similarities, open_set_identification_eval, verification_eval, verification_mul_eval
from deepface import DeepFace
from sklearn.datasets import fetch_lfw_people, fetch_olivetti_faces
import numpy as np
from .plots import save_plots
from tqdm import tqdm
from scipy.spatial.distance import cosine
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
import face_recognition
import matplotlib.pyplot as plt

DATASET = "LFW" #Dataset ot use: LFW or OLIVETTI
MIN_FACES = 4

####### Loading and parsing the dataset images #######
if DATASET == "LFW":
    lfw_people = fetch_lfw_people(color=True, min_faces_per_person=MIN_FACES, resize=1)
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

######## Defining the paths where results will be saved ######## 
SAVED_ARRAYS_PATH = f"./evaluation/saved_arrays_vgg_lfw_{MIN_FACES}" if DATASET == "LFW" else "./evaluation/saved_arrays_vgg_olivetti"
PLOTS = os.path.join(SAVED_ARRAYS_PATH, "lfw_plots") if DATASET == "LFW" else os.path.join(SAVED_ARRAYS_PATH, "olivetti_plots")
GALLERY_SET = os.path.join(SAVED_ARRAYS_PATH, "gallery_set.npy")
PROBE_SET = os.path.join(SAVED_ARRAYS_PATH, "probe_set.npy")
SIMILARITIES_PATH = os.path.join(SAVED_ARRAYS_PATH, "similarities.npy")
IDENTIFICATION_METRICS = os.path.join(SAVED_ARRAYS_PATH, "identification_metrics.csv")
VERIFICATION_METRICS = os.path.join(SAVED_ARRAYS_PATH, "verification_metrics.csv")
VERIFICATION_MUL_METRICS = os.path.join(SAVED_ARRAYS_PATH, "verification_mul_metrics.csv")

if not os.path.exists(SAVED_ARRAYS_PATH):
    os.mkdir(SAVED_ARRAYS_PATH)
if not os.path.exists(PLOTS):
    os.mkdir(PLOTS)

def get_similarity_between_two(img1, img2):
    return 1 - cosine(img1, img2)
    
######## Build the VGG Face model ########
model = DeepFace.build_model('VGG-Face')

# Localize faces and remove the unlocalized ones, only for the LFW dataset
if DATASET == "LFW" and not os.path.exists(GALLERY_SET) and not os.path.exists(PROBE_SET):
    new_X = []
    new_Y = []
    for index, template in enumerate(tqdm(X, desc="Localizing faces")):
        boxes = face_recognition.face_locations(template)
        if len(boxes) != 0:
            y_, h, w, x_ = boxes[0]
            roi = template[y_: y_+h, x_: x_+w]
            new_X.append(roi)
            new_Y.append(y[index])

    X = new_X
    y = new_Y

gallery_set = []
probe_set = []

if os.path.exists(GALLERY_SET):
    gallery_set = np.load(GALLERY_SET)
else:
    for gallery_template in tqdm(X_train, desc="Extracting gallery set feature vectors"):
        gallery_set.append(DeepFace.represent(gallery_template, model=model, detector_backend="skip"))
    np.save(GALLERY_SET, np.array(gallery_set))

if os.path.exists(PROBE_SET):
    probe_set = np.load(PROBE_SET)
else:
    for probe_template in tqdm(X_test, desc="Extracting probe set feature vectors"):
        probe_set.append(DeepFace.represent(probe_template, model=model, detector_backend="skip"))
    np.save(PROBE_SET, np.array(probe_set))

# Each element is of type (label, feature_vector)
gallery_data = np.array([(y_train[i], gallery_set[i]) for i in range(len(gallery_set))])
probe_data = np.array([(y_test[i], probe_set[i]) for i in range(len(probe_set))])

######## Load similarity matrix if present on disk ######## 
if os.path.exists(SIMILARITIES_PATH):
    all_similarities = np.load(SIMILARITIES_PATH, allow_pickle=True)
else:
    all_similarities = compute_similarities(probe_data, gallery_data, get_similarity_between_two)
    np.save(SIMILARITIES_PATH, np.array(all_similarities))

####### Load evaluation data if present - Deep Face ########
thresholds = np.arange(0, 1, 0.01)
if os.path.exists(IDENTIFICATION_METRICS) and os.path.exists(VERIFICATION_METRICS) and os.path.exists(VERIFICATION_MUL_METRICS):
    open_set_metrics = pd.read_csv(IDENTIFICATION_METRICS)
    verification_metrics = pd.read_csv(VERIFICATION_METRICS)
    verification_mul_metrics = pd.read_csv(VERIFICATION_MUL_METRICS)
else:
    ####### Compute it if not - Deep Face ######## 
    open_set_identification_metrics_by_thresholds = {}
    verification_metrics_by_thresholds = {}
    verification_mul_metrics_by_thresholds = {}
    for threshold in tqdm(thresholds, desc="TOTAL"):
        DIR, FRR, FAR, GRR = open_set_identification_eval(threshold, all_similarities=all_similarities)
        open_set_identification_metrics_by_thresholds[threshold] = [DIR, FRR, FAR, GRR]
        GAR, FRR, FAR, GRR = verification_eval(threshold, all_similarities=all_similarities)
        verification_metrics_by_thresholds[threshold] = [GAR, FRR, FAR, GRR]
        GAR, FRR, FAR, GRR = verification_mul_eval(threshold, all_similarities=all_similarities)
        verification_mul_metrics_by_thresholds[threshold] = [GAR, FRR, FAR, GRR]

    open_set_metrics = pd.DataFrame(open_set_identification_metrics_by_thresholds)
    verification_metrics = pd.DataFrame(verification_metrics_by_thresholds)
    verification_mul_metrics = pd.DataFrame(verification_mul_metrics_by_thresholds)

    #Save metrics on disk
    open_set_metrics.to_csv(IDENTIFICATION_METRICS)
    verification_metrics.to_csv(VERIFICATION_METRICS)
    verification_mul_metrics.to_csv(VERIFICATION_MUL_METRICS)

####### PLOT ######## 
save_plots("DeepFace", open_set_metrics, verification_metrics, verification_mul_metrics, thresholds, PLOTS)