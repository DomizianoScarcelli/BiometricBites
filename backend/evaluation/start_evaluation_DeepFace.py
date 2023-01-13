from .evaluation import compute_similarities, open_set_identification_eval, verification_eval, verification_mul_eval
from deepface import DeepFace
from sklearn.datasets import fetch_lfw_people, fetch_olivetti_faces
import numpy as np
from .plots import roc_auc_curve, far_frr_curve
from tqdm import tqdm
from scipy.spatial.distance import cosine
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2

DATASET = "OLIVETTI" #Dataset ot use: LFW or OLIVETTI

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

######## Defining the paths where results will be saved ######## 
SAVED_ARRAYS_PATH = "./evaluation/saved_arrays_vgg_lfw" if DATASET == "LFW" else "./evaluation/saved_arrays_vgg_olivetti"
DEEP_FACE_GALLERY_SET = os.path.join(SAVED_ARRAYS_PATH, "gallery_set.npy")
DEEP_FACE_PROBE_SET = os.path.join(SAVED_ARRAYS_PATH, "probe_set.npy")
DEEP_FACE_SIMILARITIES_PATH = os.path.join(SAVED_ARRAYS_PATH, "similarities.npy")
DEEP_FACE_IDENTIFICATION_METRICS = os.path.join(SAVED_ARRAYS_PATH, "identification_metrics.csv")
DEEP_FACE_VERIFICATION_METRICS = os.path.join(SAVED_ARRAYS_PATH, "validation_metrics.csv")
DEEP_FACE_VERIFICATION_MUL_METRICS = os.path.join(SAVED_ARRAYS_PATH, "validation_mul_metrics.csv")

if not os.path.exists(SAVED_ARRAYS_PATH):
    os.mkdir(SAVED_ARRAYS_PATH)

def get_similarity_between_two(img1, img2):
    return 1 - cosine(img1, img2)
    
######## Build feature vectors ########
model = DeepFace.build_model('VGG-Face')

gallery_set = []
probe_set = []

if os.path.exists(DEEP_FACE_GALLERY_SET):
    gallery_set = np.load(DEEP_FACE_GALLERY_SET)
else:
    for gallery_template in tqdm(X_train, desc="Extracting gallery set feature vectors"):
        gallery_set.append(DeepFace.represent(gallery_template, model=model, detector_backend="skip"))
    np.save(DEEP_FACE_GALLERY_SET, np.array(gallery_set))

if os.path.exists(DEEP_FACE_PROBE_SET):
    probe_set = np.load(DEEP_FACE_PROBE_SET)
else:
    for probe_template in tqdm(X_test, desc="Extracting probe set feature vectors"):
        probe_set.append(DeepFace.represent(probe_template, model=model, detector_backend="skip"))
    np.save(DEEP_FACE_PROBE_SET, np.array(probe_set))

# Each element is of type (label, feature_vector)
gallery_data = np.array([(y_train[i], gallery_set[i]) for i in range(len(gallery_set))])
probe_data = np.array([(y_test[i], probe_set[i]) for i in range(len(probe_set))])

######## Load similarity matrix if present on disk ######## 
if os.path.exists(DEEP_FACE_SIMILARITIES_PATH):
    all_similarities = np.load(DEEP_FACE_SIMILARITIES_PATH, allow_pickle=True)
else:
    all_similarities = compute_similarities(probe_data, gallery_data, get_similarity_between_two)
    np.save(DEEP_FACE_SIMILARITIES_PATH, np.array(all_similarities))

####### Load evaluation data if present - Deep Face ######## 
if os.path.exists(DEEP_FACE_IDENTIFICATION_METRICS) and os.path.exists(DEEP_FACE_VERIFICATION_METRICS) and os.path.exists(DEEP_FACE_VERIFICATION_MUL_METRICS):
    deep_face_open_set_metrics = pd.read_csv(DEEP_FACE_IDENTIFICATION_METRICS)
    deep_face_verification_metrics = pd.read_csv(DEEP_FACE_VERIFICATION_METRICS)
    deep_face_verification_mul_metrics = pd.read_csv(DEEP_FACE_VERIFICATION_MUL_METRICS)
else:
    ####### Compute it if not - Deep Face ######## 
    deep_face_open_set_identification_metrics_by_thresholds = {}
    deep_face_verification_metrics_by_thresholds = {}
    deep_face_verification_mul_metrics_by_thresholds = {}
    thresholds = np.arange(0, 1, 0.01)
    for threshold in tqdm(thresholds, desc="TOTAL"):
        DIR, FRR, FAR, GRR = open_set_identification_eval(threshold, all_similarities=all_similarities)
        deep_face_open_set_identification_metrics_by_thresholds[threshold] = [DIR, FRR, FAR, GRR]
        GAR, FRR, FAR, GRR = verification_eval(threshold, all_similarities=all_similarities)
        deep_face_verification_metrics_by_thresholds[threshold] = [GAR, FRR, FAR, GRR]
        GAR, FRR, FAR, GRR = verification_mul_eval(threshold, all_similarities=all_similarities)
        deep_face_verification_mul_metrics_by_thresholds[threshold] = [GAR, FRR, FAR, GRR]

    deep_face_open_set_metrics = pd.DataFrame(deep_face_open_set_identification_metrics_by_thresholds)
    deep_face_verification_metrics = pd.DataFrame(deep_face_verification_metrics_by_thresholds)
    deep_face_verification_mul_metrics = pd.DataFrame(deep_face_verification_mul_metrics_by_thresholds)

    # #Save metrics on disk (TODO: commented for now since it causes an error in the next part)
    # deep_face_open_set_metrics.to_csv(DEEP_FACE_IDENTIFICATION_METRICS)
    # deep_face_verification_metrics.to_csv(DEEP_FACE_VERIFICATION_METRICS)
    # deep_face_verification_mul_metrics.to_csv(DEEP_FACE_VERIFICATION_MUL_METRICS)

#TODO: Error because it load the numbers from the .csv files as strings and not floats
####### PLOT ######## 
deep_face_open_set_FAR_FRR = {"FAR": deep_face_open_set_metrics.iloc[2], "FRR": deep_face_open_set_metrics.iloc[1], "GAR": 1-deep_face_open_set_metrics.iloc[1]}
roc_auc_curve("openset", "DeepFace", deep_face_open_set_FAR_FRR)
far_frr_curve("openset", "DeepFace", deep_face_open_set_FAR_FRR, thresholds)

deep_face_verification_FAR_FRR = {"FAR": deep_face_verification_metrics.iloc[2], "FRR": deep_face_verification_metrics.iloc[1], "GAR": 1-deep_face_verification_metrics.iloc[1]}
roc_auc_curve("verification", "DeepFace", deep_face_verification_FAR_FRR)
far_frr_curve("verification", "DeepFace", deep_face_verification_FAR_FRR, thresholds)

deep_face_verification_mul_FAR_FRR = {"FAR": deep_face_verification_mul_metrics.iloc[2], "FRR": deep_face_verification_mul_metrics.iloc[1], "GAR": 1-deep_face_verification_mul_metrics.iloc[1]}
roc_auc_curve("verification-mul", "DeepFace", deep_face_verification_mul_FAR_FRR)
far_frr_curve("verification-mul", "DeepFace", deep_face_verification_mul_FAR_FRR, thresholds)
