"""
from .evaluation import open_set_identification_eval, verification_eval, compute_similarities
from .plots import roc_curve, far_frr_curve
from sklearn.datasets import fetch_lfw_people
from deepface import DeepFace
from tqdm import tqdm
import numpy as np
import pandas as pd
"""

from .evaluation import compute_similarities, open_set_identification_eval, verification_eval, verification_mul_eval
from deepface import DeepFace
from sklearn.datasets import fetch_lfw_people
import numpy as np
from .plots import roc_auc_curve, far_frr_curve
from tqdm import tqdm
from scipy.spatial.distance import cosine
import os
import pandas as pd

####### Loading and parsing the dataset images #######
lfw_people = fetch_lfw_people(color=True, min_faces_per_person=50, resize=0.5)
X = lfw_people.images
y = lfw_people.target
X *= 255
X = np.array(X, dtype='uint8')

######## Defining the paths where results will be saved ######## 
SAVED_ARRAYS_PATH = "./evaluation/saved_arrays"
DEEP_FACE_FEATURE_VECTORS_PATH = os.path.join(SAVED_ARRAYS_PATH, "deep_face_feature_vectors.npy")
DEEP_FACE_SIMILARITIES_PATH = os.path.join(SAVED_ARRAYS_PATH, "deep_face_similarities.npy")
DEEP_FACE_METRICS_PATH = os.path.join(SAVED_ARRAYS_PATH, "deep_face_metrics.csv")
#CLAIMS_PATH = os.path.join(SAVED_ARRAYS_PATH, "claims_number.csv")
if not os.path.exists(SAVED_ARRAYS_PATH):
    os.mkdir(SAVED_ARRAYS_PATH)

######## Build feature vectors ######## 
model = DeepFace.build_model('VGG-Face')
feature_vectors = []
if os.path.exists(DEEP_FACE_FEATURE_VECTORS_PATH):
    feature_vectors = np.load(DEEP_FACE_FEATURE_VECTORS_PATH)
else:
    feature_vectors = []
    #For loop in order to use tqdm
    for i in tqdm(range(len(lfw_people.data)), desc="Extracting feature vector"):
        feature_vectors.append(DeepFace.represent(X[i], model=model, detector_backend="skip"))
    np.save(DEEP_FACE_FEATURE_VECTORS_PATH, np.array(feature_vectors))

# Each element is of type (label, feature_vector)
data = np.array([(y[i], feature_vectors[i]) for i in range(len(lfw_people.data))])

def get_similarity_between_two(img1, img2):
    return 1 - cosine(img1, img2)

######## Load data if present on disk ######## 
if os.path.exists(DEEP_FACE_SIMILARITIES_PATH):
    all_similarities = np.load(DEEP_FACE_SIMILARITIES_PATH, allow_pickle=True)
    #claims_dict = pd.read_csv(CLAIMS_PATH)
    #genuine_claims = claims_dict.at[0, "genuine_claims"]
    #impostor_claims = claims_dict.at[0, "impostor_claims"]
else:
    all_similarities = compute_similarities(data, get_similarity_between_two)
    np.save(DEEP_FACE_SIMILARITIES_PATH, np.array(all_similarities))
    #claims_dict = {"genuine_claims": genuine_claims, "impostor_claims": impostor_claims}
    #pd.DataFrame(claims_dict, index=[0]).to_csv(CLAIMS_PATH)

######## Perform evaluation - Deep Face ######## 
deep_face_open_set_identification_metrics_by_thresholds = {}
deep_face_verification_metrics_by_thresholds = {}
deep_face_verification_mul_metrics_by_thresholds = {}
thresholds = np.arange(0, 1, 0.01) #To increase (Maybe take a step of just 0.01 to make the plots denser - only if it doesn't take too much time!)
for threshold in thresholds:
    DIR, FRR, FAR, GRR = open_set_identification_eval(threshold, all_similarities=all_similarities)
    deep_face_open_set_identification_metrics_by_thresholds[threshold] = [DIR, FRR, FAR, GRR]
    GAR, FRR, FAR, GRR = verification_eval(threshold, all_similarities=all_similarities)
    deep_face_verification_metrics_by_thresholds[threshold] = [GAR, FRR, FAR, GRR]
    GAR, FRR, FAR, GRR = verification_mul_eval(threshold, all_similarities=all_similarities)
    deep_face_verification_mul_metrics_by_thresholds[threshold] = [GAR, FRR, FAR, GRR]

#if os.path.exists(DEEP_FACE_METRICS_PATH):
#    deep_face_metrics = pd.read_csv(DEEP_FACE_METRICS_PATH)
#else:
#    deep_face_open_set_metrics = pd.DataFrame(deep_face_open_set_identification_metrics_by_thresholds)
#    deep_face_open_set_metrics.to_csv(DEEP_FACE_METRICS_PATH)

deep_face_open_set_metrics = pd.DataFrame(deep_face_open_set_identification_metrics_by_thresholds)
deep_face_open_set_FAR_FRR = {"FAR": deep_face_open_set_metrics.iloc[2], "FRR": deep_face_open_set_metrics.iloc[1], "GAR": 1-deep_face_open_set_metrics.iloc[1]}
roc_auc_curve("openset", "DeepFace", deep_face_open_set_FAR_FRR)
far_frr_curve("openset", "DeepFace", deep_face_open_set_FAR_FRR, thresholds)

deep_face_verification_metrics = pd.DataFrame(deep_face_verification_metrics_by_thresholds)
deep_face_verification_FAR_FRR = {"FAR": deep_face_verification_metrics.iloc[2], "FRR": deep_face_verification_metrics.iloc[1], "GAR": 1-deep_face_verification_metrics.iloc[1]}
roc_auc_curve("verification", "DeepFace", deep_face_verification_FAR_FRR)
far_frr_curve("verification", "DeepFace", deep_face_verification_FAR_FRR, thresholds)

deep_face_verification_mul_metrics = pd.DataFrame(deep_face_verification_mul_metrics_by_thresholds)
deep_face_verification_mul_FAR_FRR = {"FAR": deep_face_verification_mul_metrics.iloc[2], "FRR": deep_face_verification_mul_metrics.iloc[1], "GAR": 1-deep_face_verification_mul_metrics.iloc[1]}
roc_auc_curve("verification-mul", "DeepFace", deep_face_verification_mul_FAR_FRR)
far_frr_curve("verification-mul", "DeepFace", deep_face_verification_mul_FAR_FRR, thresholds)
