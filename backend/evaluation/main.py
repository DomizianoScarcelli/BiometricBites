from deepface import DeepFace
from sklearn.datasets import fetch_lfw_people
import numpy as np
import matplotlib.pyplot as plt
from .evaluation import compute_similarities, open_set_identification_eval
from tqdm import tqdm
from scipy.spatial.distance import cosine
import os
import pandas as pd

####### Loading and parsing the dataset images #######
lfw_people = fetch_lfw_people(color=True, min_faces_per_person=10, resize=0.5)
lfw_people_data = lfw_people.data
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
    for i in tqdm(range(len(lfw_people_data)), desc="Extracting feature vector"):
        feature_vectors.append(DeepFace.represent(X[i], model=model, detector_backend="skip"))
    np.save(FEATURE_VECTORS_PATH, np.array(feature_vectors))

# Each element is of type (label, feature_vector)
data = [(y[i], feature_vectors[i]) for i in range(len(lfw_people_data))]
#TODO: Don't know if this is needed, but results change if the data is sorted. data, template_list and all_similarities has to be checked better in order to see if they're correct
# data = sorted(data, key=lambda x: x[0])

def get_similarity_between_two(img1, img2):
    return 1 - cosine(img1, img2)

######## Load data if present on disk ######## 
if os.path.exists(SIMILARITIES_PATH) and os.path.exists(CLAIMS_PATH):
    all_similarities = np.load(SIMILARITIES_PATH)
    claims_dict = pd.read_csv(CLAIMS_PATH)
    genuine_claims = claims_dict.at[0, "genuine_claims"]
    impostor_claims = claims_dict.at[0, "impostor_claims"]
else:
    genuine_claims, impostor_claims, all_similarities = compute_similarities(data, get_similarity_between_two)
    np.save(SIMILARITIES_PATH, np.array(all_similarities))
    claims_dict = {"genuine_claims": genuine_claims, "impostor_claims": impostor_claims}
    pd.DataFrame(claims_dict, index=[0]).to_csv(CLAIMS_PATH)

all_similarities = [sorted(row, key=lambda x: -x[1]) for row in all_similarities] #Descending sort for similarities

def print_readable_list(list):
    for i in range(0, 100, 10):
        print("Row: ", i, list[i][:5])

# print_readable_list(data)
# print("---------------")
# print_readable_list(all_similarities)



######## Perform evaluation ######## 
open_set_identification_metrics_by_thresholds = {}
verification_metrics_by_thresholds = {}
thresholds = np.arange(0.05, 1, 0.05) #To increase (Maybe take a step of just 0.01 to make the plots denser - only if it doesn't take too much time!)

threshold = 0.8
DIR, FRR, FAR, GRR = open_set_identification_eval(data, threshold, genuine_claims=genuine_claims, impostor_claims=impostor_claims, all_similarities=all_similarities)
print(f"DIR[:5]: {DIR[:5]}, FRR: {FRR}, FAR:{FAR}, GRR: {GRR}")
# for threshold in thresholds:
#     DIR, FRR, FAR, GRR = open_set_identification_eval(data, threshold, genuine_claims=genuine_claims, impostor_claims=impostor_claims, all_similarities=all_similarities)
#     open_set_identification_metrics_by_thresholds[threshold] = [DIR, FRR, FAR, GRR]
#     

# #--------- Save metrics -----------
# DATAFRAME_PATH = os.path.join(SAVED_ARRAYS_PATH, "open_set_metrics.csv")
# open_set_metrics = pd.DataFrame(open_set_identification_metrics_by_thresholds)
# open_set_metrics.to_csv(DATAFRAME_PATH)