from deepface import DeepFace
from sklearn.datasets import fetch_lfw_people
import numpy as np
import matplotlib.pyplot as plt
lfw_people = fetch_lfw_people()

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

# for index, face_1 in enumerate(X):
#     for index_2, face_2 in enumerate(X):
#         print(f"Indexes", index, index_2)
#         if index != index_2:
#             face_1 = face_1.reshape((h,w))
#             face_2 = face_2.reshape((h,w))
#             print(DeepFace.verify(face_1.reshape((h,w)), face_2.reshape((h,w))))

plt.imshow(X[0].reshape((h, w)))
plt.show()

