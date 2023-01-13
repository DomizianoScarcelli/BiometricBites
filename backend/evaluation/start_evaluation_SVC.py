from .evaluation import open_set_identification_eval, verification_eval, verification_mul_eval, compute_similarities_svc
from sklearn.datasets import fetch_lfw_people, fetch_olivetti_faces
import numpy as np
from .plots import save_plots
from tqdm import tqdm
from scipy.spatial.distance import cosine
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
import face_recognition
import cv2

DATASET = "OLIVETTI" #Dataset ot use: LFW or OLIVETTI

####### Loading and parsing the dataset images #######
if DATASET == "LFW":
    lfw_people = fetch_lfw_people(color=True, min_faces_per_person=4, resize=0.5)
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

def represent(templates):
    feature_vectors = []
    missed_index = []
    for index, template in enumerate(tqdm(templates, desc="Extracting feature vectors")):
        if DATASET == "LFW":
            boxes = face_recognition.face_locations(template)
        else:
            boxes = [(0, 64, 64, 0)]
        encoding = face_recognition.face_encodings(template, boxes)
        if len(encoding) == 0:
            missed_index.append(index)
        else:
            feature_vectors.append(encoding[0])
    return np.array(missed_index), np.array(feature_vectors)

######## Defining the paths where results will be saved ######## 
SAVED_ARRAYS_PATH = "./evaluation/saved_arrays_svc_lfw" if DATASET == "LFW" else "./evaluation/saved_arrays_svc_olivetti"
PLOTS = os.path.join(SAVED_ARRAYS_PATH, "lfw_plots") if DATASET == "LFW" else os.path.join(SAVED_ARRAYS_PATH, "olivetti_plots")
MODEL = os.path.join(SAVED_ARRAYS_PATH, "model.pickle")
FEATURE_VECTORS_PATH = os.path.join(SAVED_ARRAYS_PATH, "feature_vectors.pickles")
PROBE_SET = os.path.join(SAVED_ARRAYS_PATH, "probe_set.npy")
SIMILARITIES_PATH = os.path.join(SAVED_ARRAYS_PATH, "similarities.npy")
IDENTIFICATION_METRICS = os.path.join(SAVED_ARRAYS_PATH, "identification_metrics.csv")
VERIFICATION_METRICS = os.path.join(SAVED_ARRAYS_PATH, "validation_metrics.csv")
VERIFICATION_MUL_METRICS = os.path.join(SAVED_ARRAYS_PATH, "validation_mul_metrics.csv")

if not os.path.exists(SAVED_ARRAYS_PATH):
    os.mkdir(SAVED_ARRAYS_PATH)
if not os.path.exists(PLOTS):
    os.mkdir(PLOTS)

def get_similarity_between_two(img1, img2):
    return 1 - cosine(img1, img2)
    
######## Build feature vectors ########
model = SVC(kernel='linear', probability=True)

if os.path.exists(FEATURE_VECTORS_PATH):
    X, y = pickle.load(open(FEATURE_VECTORS_PATH, "rb"))
else:
    missed_index, X = represent(X)
    if len(missed_index) != 0:
        y = np.delete(y, missed_index)
    pickle.dump(tuple((X,y)), open(FEATURE_VECTORS_PATH, "wb"))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

if os.path.exists(MODEL):
    model = pickle.load(open(MODEL, 'rb'))
else:
    model.fit(X_train, y_train)
    pickle.dump(model, open(MODEL, 'wb'))

# Each element is of type (label, feature_vector)
probe_data = np.array([(y_test[i], X_test[i]) for i in range(len(X_test))])

###### Load similarity matrix if present on disk ######## 
if os.path.exists(SIMILARITIES_PATH):
    all_similarities = np.load(SIMILARITIES_PATH, allow_pickle=True)
else:
    all_similarities = compute_similarities_svc(probe_data, model)
    np.save(SIMILARITIES_PATH, np.array(all_similarities))


####### Load evaluation data if present ######## 
if os.path.exists(IDENTIFICATION_METRICS) and os.path.exists(VERIFICATION_METRICS) and os.path.exists(VERIFICATION_MUL_METRICS):
    open_set_metrics = pd.read_csv(IDENTIFICATION_METRICS)
    verification_metrics = pd.read_csv(VERIFICATION_METRICS)
    verification_mul_metrics = pd.read_csv(VERIFICATION_MUL_METRICS)
else:
    ####### Compute it if not ######## 
    open_set_identification_metrics_by_thresholds = {}
    verification_metrics_by_thresholds = {}
    verification_mul_metrics_by_thresholds = {}
    thresholds = np.arange(0, 1, 0.01)
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

    #Save metrics on disk (commented for now since it gives an error)
    # open_set_metrics.to_csv(IDENTIFICATION_METRICS)
    # verification_metrics.to_csv(VERIFICATION_METRICS)
    # verification_mul_metrics.to_csv(VERIFICATION_MUL_METRICS)

#TODO: Error because it load the numbers from the .csv files as strings and not floats
####### PLOT ########
save_plots(open_set_metrics, verification_metrics, verification_mul_metrics, thresholds, PLOTS)