import os
import cv2
import pickle
import face_recognition
import numpy as np

from sklearn.svm import SVC as sklearn_SVC

from api.utils.recognition.Classifier import Classifier

class SVC(Classifier):
    def __init__(self) -> None:
        super().__init__()
        self.pickle_file_name = "face_labels_svc.pickle"
        self.classifier = sklearn_SVC(C=1, kernel='linear', probability=True)
        self.labels = None
        self.load_labels(self.pickle_file_name)
        self.load_classifier()

    def load_labels(self, pickle_file_name):
        pickle_path = os.path.join(self.labels_root, pickle_file_name)
        if not os.path.exists(pickle_path):
            with open(pickle_path, "wb") as picle_file:
                picle_file.write(pickle.dumps({}))
        with open(pickle_path, 'rb') as pickle_file:
            self.labels = pickle.load(pickle_file)
    
    def load_classifier(self):
        classifier_path = os.path.join(self.models_root, "svc_model.pickle")
        if not os.path.exists(classifier_path): return 
        with open(classifier_path, 'rb') as pickle_file:
            self.classifier = pickle.load(pickle_file)
    
    def train(self):
        self.load_classifier()
        knownEncodings = []
        knownNames = []

        for root, dirs, files in os.walk(self.image_dir):
            count = 0
            for file in files:
                print("[INFO] processing image {}/{}".format(count + 1, len(files)))
                path = os.path.join(root, file) # Save the path of each image
                name = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower() # Save the label of each image
                image = cv2.imread(path)

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                boxes = face_recognition.face_locations(img=rgb, model="hog")

                frame_encodings = face_recognition.face_encodings(face_image=rgb, known_face_locations=boxes)

                if frame_encodings:
                    knownEncodings.append(frame_encodings[0])
                    knownNames.append(name)
                count += 1
        
        print(f"Know encodings: {knownEncodings}") 

        print("Stiamo generando il tuo file di encodings..")
        data = {"encodings": np.array(knownEncodings), "names": np.array(knownNames)}

        f = open(os.path.join(self.labels_root, self.pickle_file_name), "wb")
        f.write(pickle.dumps(data))
        f.close()

        print("[INFO] start training face_encodings..")
        X = data['encodings']
        y = data['names']

        print(f"X shape: {X.shape}")

        self.classifier.fit(X, y)

        print("Saving classifier to local folder")
        f = open(os.path.join(self.models_root, "svc_model.pickle"), "wb")
        f.write(pickle.dumps(self.classifier))
        f.close()
        print("Classifier saved!")

    def recognize(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(img=rgb, model="hog")

        frame_encodings = face_recognition.face_encodings(face_image=rgb, known_face_locations=boxes)
        name = "unknown"
        confidence = 1

        if frame_encodings:
            predictions = self.classifier.predict_proba([frame_encodings[0]]).ravel()
            maxPred = np.argmax(predictions)
            confidence = predictions[maxPred]

            if confidence > 0.85:
                name = self.classifier.predict([frame_encodings[0]])[0]
            print(name)
            print(confidence)
        
                
        return frame, name, confidence
