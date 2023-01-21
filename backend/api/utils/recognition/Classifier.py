import os
import cv2
import pickle
import tensorflow as tf
import numpy as np
from PIL import Image
from bsproject.paths import SAMPLES_ROOT, LABELS_ROOT, MODELS_ROOT
from abc import ABC, abstractmethod
import math
import pandas as pd
from PIL import Image



class Classifier(ABC):
    """
    Abstract class, this shouldn't be instanced, but it describes the common fields and methods that a classifier have.
    """
    def __init__(self) -> None:
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.side_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        self.image_dir = SAMPLES_ROOT
        self.models_root = MODELS_ROOT
        self.labels_root = LABELS_ROOT
        self.image_width = 224
        self.image_height = 224
        self.create_necessary_folders()
    
    @abstractmethod
    def recognize(self, frame: str):
        pass
    
    @abstractmethod
    def train(self):
        pass

    def create_necessary_folders(self):
        """
        It creates labels and models folders if they don't already exist.
        """
        if not os.path.exists(self.labels_root):
            os.makedirs(self.labels_root)
        if not os.path.exists(self.models_root):
            os.makedirs(self.models_root)

    def is_image_preprocessed(self, file_name):
        """
        Check if the image is already preprocessed.
        """
        return "processed" in file_name
    
    def draw_label(self, frame, label, x, y):
        """
        Draw the predicted lable on the frame.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        stroke = 2
        cv2.putText(frame, label, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        return frame

    def preprocess_images(self):
        """
        Detect frontal and profile faces inside the image, crops them, applies some filters and saves them in the same directory, deleting the original images.
        If the image is already preprocessed, it is skipped.
        """
        image_width = self.image_width
        image_height = self.image_height

        # for detecting faces
        frontal_face_cascade = self.face_cascade
        side_face_cascade = self.side_face_cascade

        current_id = 0
        label_ids = {}

        # iterates through all the files in each subdirectories
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if self.is_image_preprocessed(file):
                    print(f"---Photo {file} skipped because it is already preprocessed---\n")
                    continue
                # path of the image
                path = os.path.join(root, file)

                # get the label name (name of the person)
                label = os.path.basename(root).replace(" ", ".").lower()

                # add the label (key) and its number (value)
                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1

                # load the image
                imgtest = cv2.imread(path, cv2.IMREAD_COLOR)
                imgtest = self.face_alignment(imgtest)
                try:
                    img_gray = cv2.cvtColor(imgtest, cv2.COLOR_BGR2GRAY)
                except:
                    print(f"---Photo {file} skipped because there is no face---\n")
                    continue
                image_array = np.array(imgtest, "uint8")


                # get the faces detected in the image
                frontal_faces = frontal_face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)
                profile_faces = np.array([])

                if len(frontal_faces) == 0:
                    profile_faces = side_face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)
                    if len(profile_faces) == 0:
                        print(f"---Photo {file} skipped because it doesn't contain any face---\n")
                        os.remove(path)
                        continue
                    else:
                        faces = profile_faces
                else:
                    faces = frontal_faces
                
                # save the detected face(s) and associate
                # them with the label
                for (x_, y_, w, h) in faces:
                    # resize the detected face to 224x224
                    size = (image_width, image_height)

                    # detected face region
                    roi = image_array[y_: y_ + h, x_: x_ + w]

                    # resize the detected head to target size
                    resized_image = cv2.resize(roi, size)

                    image_array = np.array(resized_image, "uint8")

                    # remove the original image
                    if os.path.exists(path): os.remove(path)

                    # replace the image with only the face
                    im = Image.fromarray(image_array)
                    new_file_name = f"{file.split('.')[0]}_processed.jpg"
                    new_path = os.path.join(root, new_file_name)
                    im.save(new_path)

    def apply_filters(self, filter, frame):
        """
        Apply different filters to increase the face features
        """
        image = tf.cast(tf.convert_to_tensor(frame), tf.uint8)
        image = tf.image.rgb_to_grayscale(image)
        
        # gray image
        if filter == 0:
            return np.asarray(image)
        # Boosting constrast
        elif filter == 1:
            contrast = tf.image.adjust_contrast(image, 0.8)
            return np.asarray(contrast)
        elif filter == 2:
            contrast = tf.image.adjust_contrast(image, 0.9)
            return np.asarray(contrast)
        elif filter == 3:
            contrast = tf.image.adjust_contrast(image, 1)
            return np.asarray(contrast)
        # Boosting brightness
        elif filter == 4:
            brightness = tf.image.adjust_brightness(image, 0.1)
            return np.asarray(brightness)
        elif filter == 5:
            brightness = tf.image.adjust_brightness(image, 0.2)
            return np.asarray(brightness)
        elif filter == 6:
            brightness = tf.image.adjust_brightness(image, 0.3)
            return np.asarray(brightness)            
                
    def detect_faces(self, frame):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        face_present = len(faces) != 0
        if not face_present:
            return frame, frame, False
        for (x_, y_, w, h) in faces:
            # draw the face detected
            cv2.rectangle(frame, (x_, y_), (x_+w, y_+h), (255, 0, 0), 2)
        (x_, y_, w, h) = faces[0]
        roi = frame[y_:y_+h, x_:x_+w]
        return frame, roi, True

    # Source: https://www.geeksforgeeks.org/face-alignment-with-opencv-and-python/
    # align face
    def face_alignment(self, frame):
        alignedFace = self.Face_Alignment(frame)
        return alignedFace 
    
    def trignometry_for_distance(self, a, b):
        return math.sqrt(((b[0] - a[0]) * (b[0] - a[0])) +\
                        ((b[1] - a[1]) * (b[1] - a[1])))
    
    # Find eyes
    def Face_Alignment(self, img):
        eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

        img_raw = img.copy()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes = eye_detector.detectMultiScale(gray_img)
    
        # for multiple people in an image find the largest 
        # pair of eyes
        if len(eyes) >= 2:
            eye = eyes[:, 2]
            container1 = []
            for i in range(0, len(eye)):
                container = (eye[i], i)
                container1.append(container)
            df = pd.DataFrame(container1, columns=[
                            "length", "idx"]).sort_values(by=['length'])
            eyes = eyes[df.idx.values[0:2]]
    
            # deciding to choose left and right eye
            eye_1 = eyes[0]
            eye_2 = eyes[1]
            if eye_1[0] > eye_2[0]:
                left_eye = eye_2
                right_eye = eye_1
            else:
                left_eye = eye_1
                right_eye = eye_2
    
            # center of eyes
            # center of right eye
            right_eye_center = (
                int(right_eye[0] + (right_eye[2]/2)), 
            int(right_eye[1] + (right_eye[3]/2)))
            right_eye_x = right_eye_center[0]
            right_eye_y = right_eye_center[1]
            cv2.circle(img, right_eye_center, 2, (255, 0, 0), 3)
    
            # center of left eye
            left_eye_center = (
                int(left_eye[0] + (left_eye[2] / 2)), 
            int(left_eye[1] + (left_eye[3] / 2)))
            left_eye_x = left_eye_center[0]
            left_eye_y = left_eye_center[1]
            cv2.circle(img, left_eye_center, 2, (255, 0, 0), 3)
    
            # finding rotation direction
            if left_eye_y > right_eye_y:
                print("Rotate image to clock direction")
                point_3rd = (right_eye_x, left_eye_y)
                direction = -1  # rotate image direction to clock
            else:
                print("Rotate to inverse clock direction")
                point_3rd = (left_eye_x, right_eye_y)
                direction = 1  # rotate inverse direction of clock
    
            cv2.circle(img, point_3rd, 2, (255, 0, 0), 2)
            a = self.trignometry_for_distance(left_eye_center, 
                                        point_3rd)
            b = self.trignometry_for_distance(right_eye_center, 
                                        point_3rd)
            c = self.trignometry_for_distance(right_eye_center, 
                                        left_eye_center)
            cos_a = (b*b + c*c - a*a)/(2*b*c)
            angle = (np.arccos(cos_a) * 180) / math.pi
    
            if direction == -1:
                angle = 90 - angle
            else:
                angle = -(90-angle)
    
            # rotate image
            new_img = Image.fromarray(img_raw)
            new_img = np.array(new_img.rotate(direction * angle))
        else:
            new_img = img
    
        return new_img