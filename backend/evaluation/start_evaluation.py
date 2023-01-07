"""
from .evaluation import open_set_identification_eval, verification_eval, compute_similarities
from .plots import roc_curve, far_frr_curve
from sklearn.datasets import fetch_lfw_people
from deepface import DeepFace
from tqdm import tqdm
import numpy as np
import pandas as pd
"""

"""
#---------Update Dataset-----------
lfw_people = fetch_lfw_people(color=True, min_faces_per_person=10)

# Introspect the images arrays to find the shapes (for plotting)
n_samples, h, w, c = lfw_people.images.shape

X = lfw_people.images
n_features = X.shape[1]

# The label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print(X.shape)
print(y.shape)

data = np.array([(y[i], X[i]) for i in range(len(lfw_people.data))])

def get_similarity_between_two(img1, img2):
    try:
        return 1 - DeepFace.verify(img1, img2)["distance"]
    except:
        return None

genuine_claims, impostor_claims, all_similarities = compute_similarities(data, get_similarity_between_two)

np.save('similarities', np.array(all_similarities)) #Save the array in order to compute it only once in a lifetime lol
print("Genuine claims", genuine_claims)
print("Impostor claims", impostor_claims)

# sim = np.load("./similarities.npy", allow_pickle=True)
"""

"""
template_list = {} #To change

open_set_identification_metrics_by_thresholds = {}
verification_metrics_by_thresholds = {}
thresholds = np.arange(0.05, 1, 0.05) #To increase (Maybe take a step of just 0.01 to make the plots denser - only if it doesn't take too much time!)

for threshold in thresholds:
    DIR, FRR, FAR, GRR = open_set_identification_eval(template_list, threshold)
    open_set_identification_metrics_by_thresholds[threshold] = [DIR, FRR, FAR, GRR]
    GAR, GRR, FAR, FRR = verification_eval(template_list, threshold)
    verification_metrics_by_thresholds[threshold] = [GAR, GRR, FAR, FRR]

#---------Plot ROC-----------
open_set_metrics = pd.DataFrame(open_set_identification_metrics_by_thresholds)
open_set_FAR_FRR = {"FAR": open_set_metrics.iloc[2], "FRR": open_set_metrics.iloc[1]}
print("ROC curve for Open Set Identification:")
roc_curve("openset", "VggFace", open_set_FAR_FRR)
print("FAR vs FRR curve for Open Set Identification:")
far_frr_curve("openset", "VggFace", open_set_FAR_FRR, thresholds)

verification_metrics = pd.DataFrame(open_set_identification_metrics_by_thresholds)
verification_FAR_FRR = {"FAR": verification_metrics.iloc[2], "FRR": verification_metrics.iloc[1]}
print("ROC curve for Verification:")
roc_curve("verification", "VggFace", verification_FAR_FRR)
print("FAR vs FRR curve for Verification:")
far_frr_curve("openset", "VggFace", open_set_FAR_FRR, thresholds)
"""

from deepface import DeepFace
from sklearn.datasets import fetch_lfw_people
import numpy as np
from .evaluation import compute_similarities, open_set_identification_eval
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
FEATURE_VECTORS_PATH = os.path.join(SAVED_ARRAYS_PATH, "feature_vectors.npy")
SIMILARITIES_PATH = os.path.join(SAVED_ARRAYS_PATH, "similarities.npy")
CLAIMS_PATH = os.path.join(SAVED_ARRAYS_PATH, "claims_number.csv")
if not os.path.exists(SAVED_ARRAYS_PATH):
    os.mkdir(SAVED_ARRAYS_PATH)
######## Build feature vectors ######## 
model = DeepFace.build_model('VGG-Face')
feature_vectors = []
if os.path.exists(FEATURE_VECTORS_PATH):
    feature_vectors = np.load(FEATURE_VECTORS_PATH)
else:
    feature_vectors = []
    #For loop in order to use tqdm
    for i in tqdm(range(len(lfw_people.data)), desc="Extracting feature vector"):
        feature_vectors.append(DeepFace.represent(X[i], model=model, detector_backend="skip"))
    np.save(FEATURE_VECTORS_PATH, np.array(feature_vectors))

# Each element is of type (label, feature_vector)
data = np.array([(y[i], feature_vectors[i]) for i in range(len(lfw_people.data))])

def get_similarity_between_two(img1, img2):
    return 1 - cosine(img1, img2)

######## Load data if present on disk ######## 
if os.path.exists(SIMILARITIES_PATH) and os.path.exists(CLAIMS_PATH):
    all_similarities = np.load(SIMILARITIES_PATH, allow_pickle=True)
    claims_dict = pd.read_csv(CLAIMS_PATH)
    genuine_claims = claims_dict.at[0, "genuine_claims"]
    impostor_claims = claims_dict.at[0, "impostor_claims"]
else:
    genuine_claims, impostor_claims, all_similarities = compute_similarities(data, get_similarity_between_two)
    np.save(SIMILARITIES_PATH, np.array(all_similarities))
    claims_dict = {"genuine_claims": genuine_claims, "impostor_claims": impostor_claims}
    pd.DataFrame(claims_dict, index=[0]).to_csv(CLAIMS_PATH)

######## Perform evaluation ######## 
open_set_identification_metrics_by_thresholds = {}
verification_metrics_by_thresholds = {}
thresholds = np.arange(0, 1, 0.01) #To increase (Maybe take a step of just 0.01 to make the plots denser - only if it doesn't take too much time!)
for threshold in thresholds:
    DIR, FRR, FAR, GRR = open_set_identification_eval(threshold, genuine_claims=len(y), impostor_claims=len(y), all_similarities=all_similarities)
    open_set_identification_metrics_by_thresholds[threshold] = [DIR, FRR, FAR, GRR]
    print(FRR, FAR)

#---------Plot ROC-----------
DATAFRAME_PATH = os.path.join(SAVED_ARRAYS_PATH, "open_set_metrics.csv")
open_set_metrics = pd.DataFrame(open_set_identification_metrics_by_thresholds)
open_set_metrics.to_csv(DATAFRAME_PATH)

#open_set_metrics = pd.DataFrame(open_set_identification_metrics_by_thresholds)
open_set_FAR_FRR = {"FAR": open_set_metrics.iloc[2], "FRR": open_set_metrics.iloc[1], "GAR": 1-open_set_metrics.iloc[1]}
#print("ROC curve for Open Set Identification:")
roc_auc_curve("openset", "VggFace", open_set_FAR_FRR)
#print("FAR vs FRR curve for Open Set Identification:")
#far_frr_curve("openset", "VggFace", open_set_FAR_FRR, thresholds)

# verification_metrics = pd.DataFrame(open_set_identification_metrics_by_thresholds)
# verification_FAR_FRR = {"FAR": verification_metrics.iloc[2], "FRR": verification_metrics.iloc[1]}
# print("ROC curve for Verification:")
# roc_curve("verification", "VggFace", verification_FAR_FRR)
# print("FAR vs FRR curve for Verification:")
# far_frr_curve("openset", "VggFace", open_set_FAR_FRR, thresholds)
