import cv2
import os
import numpy as np
from PIL import Image

from bsproject.settings import SAMPLES_ROOT

def is_image_preprocessed(file_name):
    """
    Check if the image is already preprocessed.
    """
    return "processed" in file_name

def preprocess_images():
    """
    Detect frontal and profile faces inside the image, crops them, applies some filters and saves them in the same directory, deleting the original images.
    If the image is already preprocessed, it is skipped.
    """
    image_width = 224
    image_height = 224

    # for detecting faces
    frontal_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    side_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

    current_id = 0
    label_ids = {}

    # iterates through all the files in each subdirectories
    for root, _, files in os.walk(SAMPLES_ROOT):
        for file in files:
            if is_image_preprocessed(file):
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
            img_gray = cv2.cvtColor(imgtest, cv2.COLOR_BGR2GRAY)
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
                os.remove(path)

                # replace the image with only the face
                im = Image.fromarray(image_array)
                new_file_name = f"{file.split('.')[0]}_processed.jpg"
                new_path = os.path.join(root, new_file_name)
                im.save(new_path)
