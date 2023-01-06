from .evaluation import open_set_identification_eval, verification_eval, compute_similarities
from .plots import roc_curve, far_frr_curve
from sklearn.datasets import fetch_lfw_people
from deepface import DeepFace
from tqdm import tqdm
import numpy as np
import pandas as pd

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
