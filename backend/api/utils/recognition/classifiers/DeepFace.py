from deepface import DeepFace
from ..Classifier import Classifier
import cv2
import pandas as pd

class DeepFaceClassifier(Classifier):
    def __init__(self) -> None:
        super().__init__()

    def recognize(self, frame: str):
        df: pd.DataFrame = DeepFace.find(frame, self.image_dir, enforce_detection=False, silent=True)
        for index, row in df.iterrows():
            df.at[index,"VGG-Face_cosine"] = 1 - df.at[index,"VGG-Face_cosine"]
            df.at[index, "identity"] = df.at[index, "identity"].split("/")[-2] 
        df.sort_values(by=["VGG-Face_cosine"], inplace = True, ascending=False)
        return df
    
    def get_cosine_similarity(img1, img2):
        """
        Gets cosine similarity between two images
        """
        return 1 - DeepFace.verify(img1, img2)["distance"]

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
                print(classifier.recognize(frame).to_string())

            cv2.imshow('Webcam',cv2.flip(frame, 1))
            cv2.waitKey(1)

    classifier = DeepFaceClassifier()
    test_with_cam()


    