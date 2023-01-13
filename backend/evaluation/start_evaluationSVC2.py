from .evaluation import open_set_identification_eval, verification_eval, verification_mul_eval, compute_similarities_svc
import numpy as np
from .plots import roc_auc_curve, far_frr_curve
from tqdm import tqdm
from scipy.spatial.distance import cosine
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
import face_recognition
from sklearn.datasets import fetch_olivetti_faces
import cv2

####### Loading and parsing the dataset images #######
#lfw_people = fetch_lfw_people(color=True, min_faces_per_person=10, resize=0.5)
olivetti_people = fetch_olivetti_faces()
X= olivetti_people.images
y = olivetti_people.target
X = np.array(X * 255, dtype='uint8')

def represent(templates):
    feature_vectors = []
    missed_index = []
    for template in tqdm(templates, desc="Extracting feature vectors"):
        template_RBG = cv2.cvtColor(template, cv2.COLOR_GRAY2RGB)
        encoding = face_recognition.face_encodings(template_RBG, [(0, 64, 64, 0)])
        feature_vectors.append(encoding[0])
    return np.array(missed_index), np.array(feature_vectors)

######## Defining the paths where results will be saved ######## 
SAVED_ARRAYS_PATH = "./evaluation/saved_arrays"
SVC_MODEL = os.path.join(SAVED_ARRAYS_PATH, "svc_model.pickle")
FEATURE_VECTORS_PATH = os.path.join(SAVED_ARRAYS_PATH, "feature_vectors.pickles")
SVC_PROBE_SET = os.path.join(SAVED_ARRAYS_PATH, "svc_probe_set.npy")
SVC_SIMILARITIES_PATH = os.path.join(SAVED_ARRAYS_PATH, "svc_similarities.npy")
SVC_IDENTIFICATION_METRICS = os.path.join(SAVED_ARRAYS_PATH, "svc_identification_metrics.csv")
SVC_VERIFICATION_METRICS = os.path.join(SAVED_ARRAYS_PATH, "svc_validation_metrics.csv")
SVC_VERIFICATION_MUL_METRICS = os.path.join(SAVED_ARRAYS_PATH, "svc_validation_mul_metrics.csv")

if not os.path.exists(SAVED_ARRAYS_PATH):
    os.mkdir(SAVED_ARRAYS_PATH)

def get_similarity_between_two(img1, img2):
    return 1 - cosine(img1, img2)
    
######## Build feature vectors ########
model = SVC(C=1, kernel='linear', probability=True)

if os.path.exists(FEATURE_VECTORS_PATH):
    X, y = pickle.load(open(FEATURE_VECTORS_PATH, "rb"))
else:
    missed_index, X = represent(X)
    pickle.dump(tuple((X,y)), open(FEATURE_VECTORS_PATH, "wb"))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

if os.path.exists(SVC_MODEL):
    model = pickle.load(open(SVC_MODEL, 'rb'))
else:
    model.fit(X_train, y_train)
    pickle.dump(model, open(SVC_MODEL, 'wb'))

# Each element is of type (label, feature_vector)
probe_data = np.array([(y_test[i], X_test[i]) for i in range(len(X_test))])

###### Load similarity matrix if present on disk ######## 
if os.path.exists(SVC_SIMILARITIES_PATH):
    all_similarities = np.load(SVC_SIMILARITIES_PATH, allow_pickle=True)
else:
    all_similarities = compute_similarities_svc(probe_data, model)
    np.save(SVC_SIMILARITIES_PATH, np.array(all_similarities))


####### Load evaluation data if present ######## 
if os.path.exists(SVC_IDENTIFICATION_METRICS) and os.path.exists(SVC_VERIFICATION_METRICS) and os.path.exists(SVC_VERIFICATION_MUL_METRICS):
    svc_open_set_metrics = pd.read_csv(SVC_IDENTIFICATION_METRICS)
    svc_verification_metrics = pd.read_csv(SVC_VERIFICATION_METRICS)
    svc_verification_mul_metrics = pd.read_csv(SVC_VERIFICATION_MUL_METRICS)
else:
    ####### Compute it if not ######## 
    svc_open_set_identification_metrics_by_thresholds = {}
    svc_verification_metrics_by_thresholds = {}
    svc_verification_mul_metrics_by_thresholds = {}
    thresholds = np.arange(0, 1, 0.01)
    for threshold in tqdm(thresholds, desc="TOTAL"):
        DIR, FRR, FAR, GRR = open_set_identification_eval(threshold, all_similarities=all_similarities)
        svc_open_set_identification_metrics_by_thresholds[threshold] = [DIR, FRR, FAR, GRR]
        GAR, FRR, FAR, GRR = verification_eval(threshold, all_similarities=all_similarities)
        svc_verification_metrics_by_thresholds[threshold] = [GAR, FRR, FAR, GRR]
        GAR, FRR, FAR, GRR = verification_mul_eval(threshold, all_similarities=all_similarities)
        svc_verification_mul_metrics_by_thresholds[threshold] = [GAR, FRR, FAR, GRR]

    svc_open_set_metrics = pd.DataFrame(svc_open_set_identification_metrics_by_thresholds)
    svc_verification_metrics = pd.DataFrame(svc_verification_metrics_by_thresholds)
    svc_verification_mul_metrics = pd.DataFrame(svc_verification_mul_metrics_by_thresholds)

    #Save metrics on disk
    svc_open_set_metrics.to_csv(SVC_IDENTIFICATION_METRICS)
    svc_verification_metrics.to_csv(SVC_VERIFICATION_METRICS)
    svc_verification_mul_metrics.to_csv(SVC_VERIFICATION_MUL_METRICS)

#TODO: Error because it load the numbers from the .csv files as strings and not floats
####### PLOT ########
svc_open_set_FAR_FRR = {"FAR": svc_open_set_metrics.iloc[2], "FRR": svc_open_set_metrics.iloc[1], "GAR": 1-svc_open_set_metrics.iloc[1]}
roc_auc_curve("openset", "DeepFace", svc_open_set_FAR_FRR)
far_frr_curve("openset", "DeepFace", svc_open_set_FAR_FRR, thresholds)

svc_verification_FAR_FRR = {"FAR": svc_verification_metrics.iloc[2], "FRR": svc_verification_metrics.iloc[1], "GAR": 1-svc_verification_metrics.iloc[1]}
roc_auc_curve("verification", "DeepFace", svc_verification_FAR_FRR)
far_frr_curve("verification", "DeepFace", svc_verification_FAR_FRR, thresholds)

svc_verification_mul_FAR_FRR = {"FAR": svc_verification_mul_metrics.iloc[2], "FRR": svc_verification_mul_metrics.iloc[1], "GAR": 1-svc_verification_mul_metrics.iloc[1]}
roc_auc_curve("verification-mul", "DeepFace", svc_verification_mul_FAR_FRR)
far_frr_curve("verification-mul", "DeepFace", svc_verification_mul_FAR_FRR, thresholds)