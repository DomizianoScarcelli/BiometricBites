from deepface import DeepFace
from sklearn.datasets import fetch_lfw_people
import numpy as np
import matplotlib.pyplot as plt
from .evaluation import compute_similarities
from tqdm import tqdm

#TODO: I reduced the dataset to only people with at least 10 photos, and it takes around 15 minutes execute compute_similarities
# this may be reduced using some numpy vectorization but I don't know how to do it lol. 
lfw_people = fetch_lfw_people(color=True, min_faces_per_person=10)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w, c = lfw_people.images.shape

X = lfw_people.images
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print(X.shape)
print(y.shape)


# Each element is of type (label, img_array)
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
