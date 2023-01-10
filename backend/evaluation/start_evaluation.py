from .evaluation import compute_similarities, open_set_identification_eval, verification_eval, verification_mul_eval
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC as sklearn_SVC
from sklearn.datasets import fetch_lfw_people
from .plots import roc_auc_curve, far_frr_curve
from scipy.spatial.distance import cosine
from deepface import DeepFace
import face_recognition
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

####### Loading and parsing the dataset images #######
lfw_people = fetch_lfw_people(color=True, min_faces_per_person=10, resize=0.5)
X = lfw_people.images
y = lfw_people.target
X = np.array(X*255, dtype='uint8')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

######## Defining the paths where results will be saved ######## 
SAVED_ARRAYS_PATH = "./evaluation/saved_arrays"
DEEP_FACE_GALLERY_SET = os.path.join(SAVED_ARRAYS_PATH, "deep_face_gallery_set.npy")
DEEP_FACE_PROBE_SET = os.path.join(SAVED_ARRAYS_PATH, "deep_face_probe_set.npy")
DEEP_FACE_SIMILARITIES_PATH = os.path.join(SAVED_ARRAYS_PATH, "deep_face_similarities.npy")
DEEP_FACE_IDENTIFICATION_METRICS = os.path.join(SAVED_ARRAYS_PATH, "deep_face_identification_metrics.csv")
DEEP_FACE_VERIFICATION_METRICS = os.path.join(SAVED_ARRAYS_PATH, "deep_face_validation_metrics.csv")
DEEP_FACE_VERIFICATION_MUL_METRICS = os.path.join(SAVED_ARRAYS_PATH, "deep_face_validation_mul_metrics.csv")
SVC_ENCODINGS_PATH = os.path.join(SAVED_ARRAYS_PATH, "svc_encodings.npy")
SVC_LABELS = os.path.join(SAVED_ARRAYS_PATH, "svc_labels.npy")
SVC_SIMILARITIES_PATH = os.path.join(SAVED_ARRAYS_PATH, "svc_similarities.npy")
SVC_IDENTIFICATION_METRICS = os.path.join(SAVED_ARRAYS_PATH, "svc_identification_metrics.csv")
SVC_VERIFICATION_METRICS = os.path.join(SAVED_ARRAYS_PATH, "svc_validation_metrics.csv")
SVC_VERIFICATION_MUL_METRICS = os.path.join(SAVED_ARRAYS_PATH, "svc_validation_mul_metrics.csv")

if not os.path.exists(SAVED_ARRAYS_PATH):
    os.mkdir(SAVED_ARRAYS_PATH)

def get_similarity_between_two(img1, img2):
    return 1 - cosine(img1, img2)

def evaluation_deepface():
    ######## Build feature vectors ########
    model = DeepFace.build_model('VGG-Face')

    gallery_set = []
    probe_set = []

    if os.path.exists(DEEP_FACE_GALLERY_SET):
        gallery_set = np.load(DEEP_FACE_GALLERY_SET)
    else:
        for gallery_template in tqdm(X_train, desc="Extracting gallery set feature vectors"):
            gallery_set.append(DeepFace.represent(gallery_template, model=model, detector_backend="skip"))
        np.save(DEEP_FACE_GALLERY_SET, np.array(gallery_set))

    if os.path.exists(DEEP_FACE_PROBE_SET):
        probe_set = np.load(DEEP_FACE_PROBE_SET)
    else:
        for probe_template in tqdm(X_test, desc="Extracting probe set feature vectors"):
            probe_set.append(DeepFace.represent(probe_template, model=model, detector_backend="skip"))
        np.save(DEEP_FACE_PROBE_SET, np.array(probe_set))

    # Each element is of type (label, feature_vector)
    gallery_data = np.array([(y_train[i], gallery_set[i]) for i in range(len(gallery_set))])
    probe_data = np.array([(y_test[i], probe_set[i]) for i in range(len(probe_set))])

    ######## Load similarity matrix if present on disk ######## 
    if os.path.exists(DEEP_FACE_SIMILARITIES_PATH):
        all_similarities = np.load(DEEP_FACE_SIMILARITIES_PATH, allow_pickle=True)
    else:
        all_similarities = compute_similarities(probe_data, gallery_data, get_similarity_between_two)
        np.save(DEEP_FACE_SIMILARITIES_PATH, np.array(all_similarities))

    ####### Load evaluation data if present - Deep Face ######## 
    if os.path.exists(DEEP_FACE_IDENTIFICATION_METRICS) and os.path.exists(DEEP_FACE_VERIFICATION_METRICS) and os.path.exists(DEEP_FACE_VERIFICATION_MUL_METRICS):
        deep_face_open_set_metrics = pd.read_csv(DEEP_FACE_IDENTIFICATION_METRICS)
        deep_face_verification_metrics = pd.read_csv(DEEP_FACE_VERIFICATION_METRICS)
        deep_face_verification_mul_metrics = pd.read_csv(DEEP_FACE_VERIFICATION_MUL_METRICS)
    else:
        ####### Compute it if not - Deep Face ######## 
        deep_face_open_set_identification_metrics_by_thresholds = {}
        deep_face_verification_metrics_by_thresholds = {}
        deep_face_verification_mul_metrics_by_thresholds = {}
        thresholds = np.arange(0, 1, 0.01)
        for threshold in tqdm(thresholds, desc="TOTAL"):
            DIR, FRR, FAR, GRR = open_set_identification_eval(threshold, all_similarities=all_similarities)
            deep_face_open_set_identification_metrics_by_thresholds[threshold] = [DIR, FRR, FAR, GRR]
            GAR, FRR, FAR, GRR = verification_eval(threshold, all_similarities=all_similarities)
            deep_face_verification_metrics_by_thresholds[threshold] = [GAR, FRR, FAR, GRR]
            GAR, FRR, FAR, GRR = verification_mul_eval(threshold, all_similarities=all_similarities)
            deep_face_verification_mul_metrics_by_thresholds[threshold] = [GAR, FRR, FAR, GRR]

        deep_face_open_set_metrics = pd.DataFrame(deep_face_open_set_identification_metrics_by_thresholds)
        deep_face_verification_metrics = pd.DataFrame(deep_face_verification_metrics_by_thresholds)
        deep_face_verification_mul_metrics = pd.DataFrame(deep_face_verification_mul_metrics_by_thresholds)

        #Save metrics on disk
        deep_face_open_set_metrics.to_csv(DEEP_FACE_IDENTIFICATION_METRICS)
        deep_face_verification_metrics.to_csv(DEEP_FACE_VERIFICATION_METRICS)
        deep_face_verification_mul_metrics.to_csv(DEEP_FACE_VERIFICATION_MUL_METRICS)


        ####### PLOT ########
        deep_face_open_set_FAR_FRR = {"FAR": deep_face_open_set_metrics.iloc[2], "FRR": deep_face_open_set_metrics.iloc[1], "GAR": 1-deep_face_open_set_metrics.iloc[1]}
        roc_auc_curve("openset", "DeepFace", deep_face_open_set_FAR_FRR)
        far_frr_curve("openset", "DeepFace", deep_face_open_set_FAR_FRR, thresholds)

        deep_face_verification_FAR_FRR = {"FAR": deep_face_verification_metrics.iloc[2], "FRR": deep_face_verification_metrics.iloc[1], "GAR": 1-deep_face_verification_metrics.iloc[1]}
        roc_auc_curve("verification", "DeepFace", deep_face_verification_FAR_FRR)
        far_frr_curve("verification", "DeepFace", deep_face_verification_FAR_FRR, thresholds)

        deep_face_verification_mul_FAR_FRR = {"FAR": deep_face_verification_mul_metrics.iloc[2], "FRR": deep_face_verification_mul_metrics.iloc[1], "GAR": 1-deep_face_verification_mul_metrics.iloc[1]}
        roc_auc_curve("verification-mul", "DeepFace", deep_face_verification_mul_FAR_FRR)
        far_frr_curve("verification-mul", "DeepFace", deep_face_verification_mul_FAR_FRR, thresholds)

def evaluation_svc():
    def svc_compute_similarities(classifier):
        all_similarities = []
        for index, test_image in enumerate(tqdm(X_test, "SVC: Computing similarities..")):
            label_i = y_test[index]
            boxes = face_recognition.face_locations(img=test_image, model="hog")
            image_encoding = face_recognition.face_encodings(face_image=test_image, known_face_locations=boxes)
            if image_encoding:
                row_similarities = []
                probabilities = classifier.predict_proba(image_encoding)
                for label_j, probability in enumerate(probabilities[0]):
                    row_similarities.append(np.array([label_j, probability]))
                all_similarities.append(np.array([label_i, row_similarities]))
        return all_similarities

    #Train the model using the training set (gallery set)
    svc = sklearn_SVC(C=1, kernel='linear', probability=True)
    train_images = []
    labels = []
    if os.path.exists(SVC_ENCODINGS_PATH) and os.path.exists(SVC_LABELS):
        train_images = np.load(SVC_ENCODINGS_PATH, allow_pickle=True)
        labels = np.load(SVC_LABELS, allow_pickle=True)
    else:
        for index, train_image in enumerate(tqdm(X, "SVC: Encoding the train images..")):
            boxes = face_recognition.face_locations(img=train_image, model="hog")
            image_encoding = face_recognition.face_encodings(face_image=train_image, known_face_locations=boxes)
            if image_encoding:
                train_images.append(image_encoding[0])
                labels.append(y[index])
        np.save(SVC_ENCODINGS_PATH, np.array(train_images))
        np.save(SVC_LABELS, np.array(labels))
    svc.fit(train_images, labels)

    ######## Load similarity matrix if present on disk ######## 
    all_similarities = None
    if os.path.exists(SVC_SIMILARITIES_PATH):
        all_similarities = np.load(SVC_SIMILARITIES_PATH, allow_pickle=True)
    else:
        all_similarities = svc_compute_similarities(svc)
        np.save(SVC_SIMILARITIES_PATH, np.array(all_similarities))

    svc_open_set_identification_metrics_by_thresholds = {}
    svc_verification_mul_metrics_by_thresholds = {}
    thresholds = np.arange(0, 1, 0.01)
    if os.path.exists(DEEP_FACE_IDENTIFICATION_METRICS) and os.path.exists(DEEP_FACE_VERIFICATION_MUL_METRICS):
        svc_open_set_metrics = pd.read_csv(SVC_IDENTIFICATION_METRICS)
        svc_verification_mul_metrics = pd.read_csv(SVC_VERIFICATION_MUL_METRICS)
    else:
        for threshold in tqdm(thresholds, "SVC: Computing metrics.."):
            DIR, FRR, FAR, GRR = open_set_identification_eval(threshold, all_similarities=all_similarities)
            svc_open_set_identification_metrics_by_thresholds[threshold] = [DIR, FRR, FAR, GRR]
            GAR, FRR, FAR, GRR = verification_mul_eval(threshold, all_similarities=all_similarities)
            svc_verification_mul_metrics_by_thresholds[threshold] = [GAR, FRR, FAR, GRR]

        svc_open_set_metrics = pd.DataFrame(svc_open_set_identification_metrics_by_thresholds)
        svc_verification_mul_metrics = pd.DataFrame(svc_verification_mul_metrics_by_thresholds)

        #Save metrics on disk
        svc_open_set_metrics.to_csv(SVC_IDENTIFICATION_METRICS)
        svc_verification_mul_metrics.to_csv(SVC_VERIFICATION_MUL_METRICS)

        ####### PLOT ########
        svc_open_set_FAR_FRR = {"FAR": svc_open_set_metrics.iloc[2], "FRR": svc_open_set_metrics.iloc[1], "GAR": 1-svc_open_set_metrics.iloc[1]}
        roc_auc_curve("openset", "SVC", svc_open_set_FAR_FRR)
        far_frr_curve("openset", "SVC", svc_open_set_FAR_FRR, thresholds)

        svc_verification_mul_FAR_FRR = {"FAR": svc_verification_mul_metrics.iloc[2], "FRR": svc_verification_mul_metrics.iloc[1], "GAR": 1-svc_verification_mul_metrics.iloc[1]}
        roc_auc_curve("verification-mul", "SVC", svc_verification_mul_FAR_FRR)
        far_frr_curve("verification-mul", "SVC", svc_verification_mul_FAR_FRR, thresholds)

evaluation_svc()