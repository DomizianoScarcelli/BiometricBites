from .evaluation import compute_similarities, open_set_identification_eval, verification_eval, verification_mul_eval
from sklearn.datasets import fetch_lfw_people, fetch_olivetti_faces
import numpy as np
from .plots import save_plots
from tqdm import tqdm
from scipy.spatial.distance import cosine
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from scipy.stats import pearsonr

def extract_histogram(img):
    tmp_model = cv2.face.LBPHFaceRecognizer_create(
                    radius = 1, # The radius used for building the Circular Local Binary Pattern. The greater the radius, the smoother the image but more spatial information you can get
                    neighbors = 8, # The number of sample points to build a Circular Local Binary Pattern. An appropriate value is to use 8 sample points. Keep in mind: the more sample points you include, the higher the computational cost
                    grid_x = 8, # The number of cells in the horizontal direction, 8 is a common value used in publications. The more cells, the finer the grid, the higher the dimensionality of the resulting feature vector
                    grid_y = 8, # The number of cells in the vertical direction, 8 is a common value used in publications. The more cells, the finer the grid, the higher the dimensionality of the resulting feature vector
                )  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tmp_model.train([img], np.array([0]))
    histogram = tmp_model.getHistograms()[0][0]

    return histogram / max(histogram) # return normalized histogram

def get_correlation_between_two(hist1, hist2):
    return pearsonr(hist1, hist2)[0]

DATASET = "LFW" #Dataset ot use: LFW or OLIVETTI

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

######## Defining the paths where results will be saved ######## 
SAVED_ARRAYS_PATH = "backend/evaluation/saved_arrays_lbph_lfw" if DATASET == "LFW" else "./evaluation/saved_arrays_lbph_olivetti"
PLOTS = os.path.join(SAVED_ARRAYS_PATH, "lfw_plots") if DATASET == "LFW" else os.path.join(SAVED_ARRAYS_PATH, "olivetti_plots")
GALLERY_SET = os.path.join(SAVED_ARRAYS_PATH, "gallery_set.npy")
GALLERY_LABEL = os.path.join(SAVED_ARRAYS_PATH, "gallery_label.npy")
PROBE_SET = os.path.join(SAVED_ARRAYS_PATH, "probe_set.npy")
PROBE_LABEL = os.path.join(SAVED_ARRAYS_PATH, "probe_label.npy")
SIMILARITIES_PATH = os.path.join(SAVED_ARRAYS_PATH, "similarities.npy")
IDENTIFICATION_METRICS = os.path.join(SAVED_ARRAYS_PATH, "identification_metrics.csv")
VERIFICATION_METRICS = os.path.join(SAVED_ARRAYS_PATH, "verification_metrics.csv")
VERIFICATION_MUL_METRICS = os.path.join(SAVED_ARRAYS_PATH, "verification_mul_metrics.csv")

def apply_filters(filter, template):
        """
        Apply different filters to increase the face features
        """
        image = tf.cast(tf.convert_to_tensor(template), tf.uint8)
        
        # Boosting constrast
        if filter == 0:
            contrast = tf.image.adjust_contrast(image, 0.8)
            return np.array(contrast* 255, dtype='uint8') 
        elif filter == 1:
            contrast = tf.image.adjust_contrast(image, 0.9)
            return np.array(contrast* 255, dtype='uint8') 
        elif filter == 2:
            contrast = tf.image.adjust_contrast(image, 1)
            return np.array(contrast* 255, dtype='uint8')
        # Boosting brightness        
        elif filter == 3:
            brightness = tf.image.adjust_brightness(image, 0.1)
            return np.array(brightness* 255, dtype='uint8')
        elif filter == 4:
            brightness = tf.image.adjust_brightness(image, 0.2)
            return np.array(brightness* 255, dtype='uint8')
        elif filter == 5:
            brightness = tf.image.adjust_brightness(image, 0.3)
            return np.array(brightness* 255, dtype='uint8')  

if not os.path.exists(SAVED_ARRAYS_PATH):
    os.mkdir(SAVED_ARRAYS_PATH)
if not os.path.exists(PLOTS):
    os.mkdir(PLOTS)
    
######## Build feature vectors ########
gallery_set = []
probe_set = []
galery_label = []
probe_label = []

if os.path.exists(GALLERY_SET):
    gallery_set = np.load(GALLERY_SET)
    galery_label = np.load(GALLERY_LABEL)
else:
    for index,gallery_template in enumerate(tqdm(X_train, desc="Extracting gallery set feature vectors")):
        gallery_set.append(extract_histogram(gallery_template))
        
        #apply filters
        for i in range(6):
            template=apply_filters(i,gallery_template)
            gallery_set.append(extract_histogram(template))
            galery_label.append(y_train[index])
    np.save(GALLERY_SET, np.array(gallery_set))
    np.save(GALLERY_LABEL, np.array(galery_label))

if os.path.exists(PROBE_SET):
    probe_set = np.load(PROBE_SET)
    probe_label = np.load(PROBE_LABEL)
else:
    for index,probe_template in enumerate(tqdm(X_test, desc="Extracting probe set feature vectors")):
        probe_set.append(extract_histogram(probe_template))
        
        #apply filters
        for i in range(6):
            template=apply_filters(i,probe_template)
            probe_set.append(extract_histogram(template))
            probe_label.append(y_test[index])
    np.save(PROBE_SET, np.array(probe_set))
    np.save(PROBE_LABEL, np.array(probe_label))

# Each element is of type (label, feature_vector)
gallery_data = np.array([(galery_label[i], gallery_set[i]) for i in range(len(gallery_set))])
probe_data = np.array([(probe_label[i], probe_set[i]) for i in range(len(probe_set))])

######## Load similarity matrix if present on disk ######## 
if os.path.exists(SIMILARITIES_PATH):
    all_similarities = np.load(SIMILARITIES_PATH, allow_pickle=True)
else:
    all_similarities = compute_similarities(probe_data, gallery_data, get_correlation_between_two)
    np.save(SIMILARITIES_PATH, np.array(all_similarities))

####### Load evaluation data if present - LBPH ########
thresholds = np.arange(0, 1, 0.01)
if os.path.exists(IDENTIFICATION_METRICS) and os.path.exists(VERIFICATION_METRICS) and os.path.exists(VERIFICATION_MUL_METRICS):
    open_set_metrics = pd.read_csv(IDENTIFICATION_METRICS)
    verification_metrics = pd.read_csv(VERIFICATION_METRICS)
    verification_mul_metrics = pd.read_csv(VERIFICATION_MUL_METRICS)
else:
    ####### Compute it if not - LBPH ######## 
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
# TODO: in plots.py creare i plot per LBPH (tips: vedi quelli utilizzati da DeepFace)
save_plots("LBPH", open_set_metrics, verification_metrics, verification_mul_metrics, thresholds, PLOTS)