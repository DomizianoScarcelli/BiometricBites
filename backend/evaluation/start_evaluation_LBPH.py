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
    from scipy.stats import pearsonr, spearmanr, kendalltau  
    return pearsonr(hist1, hist2)[0]

####### Loading and parsing the dataset images #######
olivetti_people = fetch_olivetti_faces()
X = olivetti_people.images
y = olivetti_people.target
X = np.array(X*255, dtype='uint8')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

####### TEST (DA TOGLIERE NELLA VERSIONE FINALE) #######

# for i in range(0, len(X)):
#     for j in range(0, len(X)):
#         if y[i] == y[j]:    
#             img_i = X[i]
#             img_j = X[j]

#             # img_i = detect_face(X[i])
#             # img_j = detect_face(X[j])

#             print(get_correlation_between_two(extract_histogram(img_i), extract_histogram(img_j)))
            
#             fig = plt.figure()
 
#             ax1 = fig.add_subplot(1,2,1)
#             plt.imshow(img_i)
#             plt.axis('off')
            
#             ax2 = fig.add_subplot(1,2,2)
#             plt.imshow(img_j)
#             plt.axis('off')
            
#             plt.show()

##############

# ######## Defining the paths where results will be saved ######## 
# SAVED_ARRAYS_PATH = "./evaluation/saved_arrays"
# DEEP_FACE_GALLERY_SET = os.path.join(SAVED_ARRAYS_PATH, "deep_face_gallery_set.npy")
# DEEP_FACE_PROBE_SET = os.path.join(SAVED_ARRAYS_PATH, "deep_face_probe_set.npy")
# DEEP_FACE_SIMILARITIES_PATH = os.path.join(SAVED_ARRAYS_PATH, "deep_face_similarities.npy")
# DEEP_FACE_IDENTIFICATION_METRICS = os.path.join(SAVED_ARRAYS_PATH, "deep_face_identification_metrics.csv")
# DEEP_FACE_VERIFICATION_METRICS = os.path.join(SAVED_ARRAYS_PATH, "deep_face_validation_metrics.csv")
# DEEP_FACE_VERIFICATION_MUL_METRICS = os.path.join(SAVED_ARRAYS_PATH, "deep_face_validation_mul_metrics.csv")

# if not os.path.exists(SAVED_ARRAYS_PATH):
#     os.mkdir(SAVED_ARRAYS_PATH)

# ######## Build feature vectors ########
# gallery_set = []
# probe_set = []

# if os.path.exists(DEEP_FACE_GALLERY_SET):
#     gallery_set = np.load(DEEP_FACE_GALLERY_SET)
# else:
#     for gallery_template in tqdm(X_train, desc="Extracting gallery set feature vectors"):
#         gallery_set.append(extract_histogram(gallery_template))
#     np.save(DEEP_FACE_GALLERY_SET, np.array(gallery_set))

# if os.path.exists(DEEP_FACE_PROBE_SET):
#     probe_set = np.load(DEEP_FACE_PROBE_SET)
# else:
#     for probe_template in tqdm(X_test, desc="Extracting probe set feature vectors"):
#         probe_set.append(extract_histogram(probe_template))
#     np.save(DEEP_FACE_PROBE_SET, np.array(probe_set))

# # Each element is of type (label, feature_vector)
# gallery_data = np.array([(y_train[i], gallery_set[i]) for i in range(len(gallery_set))])
# probe_data = np.array([(y_test[i], probe_set[i]) for i in range(len(probe_set))])

# ######## Load similarity matrix if present on disk ######## 
# if os.path.exists(DEEP_FACE_SIMILARITIES_PATH):
#     all_similarities = np.load(DEEP_FACE_SIMILARITIES_PATH, allow_pickle=True)
# else:
#     all_similarities = compute_similarities(probe_data, gallery_data, get_correlation_between_two)
#     np.save(DEEP_FACE_SIMILARITIES_PATH, np.array(all_similarities))

# ####### Load evaluation data if present - Deep Face ######## 
# if os.path.exists(DEEP_FACE_IDENTIFICATION_METRICS) and os.path.exists(DEEP_FACE_VERIFICATION_METRICS) and os.path.exists(DEEP_FACE_VERIFICATION_MUL_METRICS):
#     deep_face_open_set_metrics = pd.read_csv(DEEP_FACE_IDENTIFICATION_METRICS)
#     deep_face_verification_metrics = pd.read_csv(DEEP_FACE_VERIFICATION_METRICS)
#     deep_face_verification_mul_metrics = pd.read_csv(DEEP_FACE_VERIFICATION_MUL_METRICS)
# else:
#     ####### Compute it if not - Deep Face ######## 
#     deep_face_open_set_identification_metrics_by_thresholds = {}
#     deep_face_verification_metrics_by_thresholds = {}
#     deep_face_verification_mul_metrics_by_thresholds = {}
#     thresholds = np.arange(0, 1, 0.01)
#     for threshold in tqdm(thresholds, desc="TOTAL"):
#         DIR, FRR, FAR, GRR = open_set_identification_eval(threshold, all_similarities=all_similarities)
#         deep_face_open_set_identification_metrics_by_thresholds[threshold] = [DIR, FRR, FAR, GRR]
#         GAR, FRR, FAR, GRR = verification_eval(threshold, all_similarities=all_similarities)
#         deep_face_verification_metrics_by_thresholds[threshold] = [GAR, FRR, FAR, GRR]
#         GAR, FRR, FAR, GRR = verification_mul_eval(threshold, all_similarities=all_similarities)
#         deep_face_verification_mul_metrics_by_thresholds[threshold] = [GAR, FRR, FAR, GRR]

#     deep_face_open_set_metrics = pd.DataFrame(deep_face_open_set_identification_metrics_by_thresholds)
#     deep_face_verification_metrics = pd.DataFrame(deep_face_verification_metrics_by_thresholds)
#     deep_face_verification_mul_metrics = pd.DataFrame(deep_face_verification_mul_metrics_by_thresholds)

#     #Save metrics on disk
#     deep_face_open_set_metrics.to_csv(DEEP_FACE_IDENTIFICATION_METRICS)
#     deep_face_verification_metrics.to_csv(DEEP_FACE_VERIFICATION_METRICS)
#     deep_face_verification_mul_metrics.to_csv(DEEP_FACE_VERIFICATION_MUL_METRICS)


# ####### PLOT ########
# deep_face_open_set_FAR_FRR = {"FAR": deep_face_open_set_metrics.iloc[2], "FRR": deep_face_open_set_metrics.iloc[1], "GAR": 1-deep_face_open_set_metrics.iloc[1]}
# roc_auc_curve("openset", "DeepFace", deep_face_open_set_FAR_FRR)
# far_frr_curve("openset", "DeepFace", deep_face_open_set_FAR_FRR, thresholds)

# deep_face_verification_FAR_FRR = {"FAR": deep_face_verification_metrics.iloc[2], "FRR": deep_face_verification_metrics.iloc[1], "GAR": 1-deep_face_verification_metrics.iloc[1]}
# roc_auc_curve("verification", "DeepFace", deep_face_verification_FAR_FRR)
# far_frr_curve("verification", "DeepFace", deep_face_verification_FAR_FRR, thresholds)

# deep_face_verification_mul_FAR_FRR = {"FAR": deep_face_verification_mul_metrics.iloc[2], "FRR": deep_face_verification_mul_metrics.iloc[1], "GAR": 1-deep_face_verification_mul_metrics.iloc[1]}
# roc_auc_curve("verification-mul", "DeepFace", deep_face_verification_mul_FAR_FRR)
# far_frr_curve("verification-mul", "DeepFace", deep_face_verification_mul_FAR_FRR, thresholds)
