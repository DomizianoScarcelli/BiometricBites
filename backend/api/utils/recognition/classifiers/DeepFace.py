from deepface import DeepFace
from ..Classifier import Classifier
import cv2
import pandas as pd
from scipy.spatial.distance import cosine
import os
import numpy as np

class DeepFaceClassifier(Classifier):
    def __init__(self) -> None:
        super().__init__()
        self.GALLERY_PATH = os.path.join(self.models_root, "vggface_gallery.npy")
        self.THRESHOLD = 0.8
        self.gallery = self.load_gallery()
        self.model = DeepFace.build_model("VGG-Face")

    def load_gallery(self):
        if os.path.exists(self.GALLERY_PATH):
            return np.load(self.GALLERY_PATH, allow_pickle=True)
        return np.array([])

    def build_gallery(self):
        gallery = []
        for root, dirs, files in os.walk(self.image_dir):
            for file in files:
                path = os.path.join(root, file) # Save the path of each image
                name = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower() # Save the label of each image
                image = cv2.imread(path)
                #TODO: in this way, each time the whole gallery is converted to feature vectors, and not only the newest photos
                try:
                    feature_vector = DeepFace.represent(image)
                    gallery.append((name, feature_vector))
                except:
                    pass
        self.gallery = np.array(gallery)
        np.save(self.GALLERY_PATH, self.gallery)

    def train(self):
        self.preprocess_images()
        self.build_gallery()
        
    def recognize(self, frame: str):
        similarities = []
        roi, face_present = self.detect_faces(frame)
        if not face_present: return None
        probe_feature_vector = DeepFace.represent(roi, model=self.model, detector_backend='skip')

        for (label, feature_vector) in self.gallery:
            similarity = 1 - cosine(feature_vector, probe_feature_vector)
            similarities.append((label, similarity))
        best_label, best_similarity = max(similarities, key=lambda x: x[1])
        if best_similarity >= self.THRESHOLD:
            return best_label
        return "Unknown"

if __name__ == "__main__":
    def test_with_cam():
        DELTA_RECOGNIZE = 5
        counter = 0
        cap = cv2.VideoCapture(0)
        while True:
            counter += 1
            success, frame = cap.read()
            if counter % DELTA_RECOGNIZE == 0:
                counter = 0
                print(classifier.recognize(frame))

            cv2.imshow('Webcam',cv2.flip(frame, 1))
            cv2.waitKey(1)

    classifier = DeepFaceClassifier()
    classifier.train()
    test_with_cam()


    