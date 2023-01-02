import os
import cv2
import pickle
import face_recognition
import numpy as np
from api.utils.recognition.Classifier import Classifier

from PIL import Image

from keras_preprocessing.image import img_to_array
from keras_vggface import utils
from keras.models import load_model
from keras.layers import Dense, GlobalAveragePooling2D, Convolution2D, Flatten, MaxPooling2D, Dropout
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras_vggface.vggface import VGGFace

class VGGFACE(Classifier):
    def __init__(self) -> None:
        super().__init__()
        self.side_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        self.image_width = 224
        self.image_height = 224
        self.model = None
        self.labels = None
        self.gallery = None
        self.pickle_file_name = "face_labels_vggface.pickle"
        
    def load_labels_from_disk(self):
        label_path = os.path.join(self.labels_root, self.pickle_file_name)
        if not os.path.exists(label_path):
            class_dictionary = {}
        else:
            with open(label_path, "rb") as f: 
                class_dictionary = pickle.load(f)
        return [value for _, value in class_dictionary.items()]

        
    def is_image_preprocessed(self, file_name):
        """
        Check if the image is already preprocessed.
        """
        return "processed" in file_name

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
                label = os.path.basename(root).replace(" ", "-").lower()

                # add the label (key) and its number (value)
                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1

                # load the image
                imgtest = cv2.imread(path, cv2.IMREAD_COLOR)
                try:
                    img_gray = cv2.cvtColor(imgtest, cv2.COLOR_BGR2GRAY)
                except: 
                    print(f"---Photo {file} skipped because the path wasn't found ---\n")
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
                    if os.path.exists(path):
                        os.remove(path)

                    # replace the image with only the face
                    im = Image.fromarray(image_array)
                    new_file_name = f"{file.split('.')[0]}_processed.jpg"
                    new_path = os.path.join(root, new_file_name)
                    im.save(new_path)
            
        
    def train(self):
        """
        Train the model with the new images
        """
        # #TODO: this can be done without the train generator, but while encoding the files in the os.walk 
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        train_generator = train_datagen.flow_from_directory(
            self.image_dir,
            target_size=(224,224),
            color_mode='rgb',
            batch_size=32,
            class_mode='categorical',
            shuffle=True)

        self.model = VGGFace(include_top=True, model='vgg16', input_shape=(224, 224, 3))

        class_dictionary_og = train_generator.class_indices

        class_dictionary = { value:key for key, value in class_dictionary_og.items() } #This has the structure {0: name1, 1: name2, 2: name3}

        # save the class dictionary on disk with pickle
        face_label_filename = os.path.join(self.labels_root, self.pickle_file_name)
        with open(face_label_filename, 'wb') as f: 
            pickle.dump(class_dictionary, f)

        #Save to a dictionary the mapping between the identities and the encodings
        encodings_dictionary = {}

        for root, dirs, files in os.walk(self.image_dir):
            for file in files:
                name = os.path.split(os.path.split(os.path.join(root, file))[-2])[-1]
                if name not in class_dictionary_og:
                    continue
                identifier = class_dictionary_og[name]
                if identifier not in encodings_dictionary:
                    encodings_dictionary[identifier] = []
                else:
                    encodings_dictionary[identifier].append(self.encode(cv2.imread(os.path.join(root, file))))

        self.labels = class_dictionary
        self.gallery = encodings_dictionary
    
    def encode(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except:
            return np.array([])
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        image_array = np.array(frame, "uint8")

        if len(faces) != 1:
            return np.array([])

        x_, y_, w, h = faces[0]
        # resize the detected face to 224x224
        size = (self.image_width, self.image_height)
        roi = image_array[y_: y_ + h, x_: x_ + w]
        resized_image = cv2.resize(roi, size)

        # prepare the image for prediction
        x = img_to_array(resized_image)
        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_input(x, version=1)

        encoding = self.model.predict(x)[0]

        return np.array(encoding).flatten()


    def recognize(self, frame: str):
        def cosine_similarity(encoding_1, encoding_2):
            a = np.matmul(np.transpose(encoding_1), encoding_2)
            b = np.sum(np.multiply(encoding_2, encoding_2))
            c = np.sum(np.multiply(encoding_1, encoding_1))
            return (a / (np.sqrt(b) * np.sqrt(c)))
        

        source_image_encoding = self.encode(frame)
        
        if len(source_image_encoding) == 0:
            return []

        # Calculate cosine similarity from source image to all the other images in the gallery
        similarity_array = [None for _ in range(len(self.gallery))]
        for identifier, images in self.gallery.items():
            max_similarity = 0
            for image in images:
                try:
                    similarity = cosine_similarity(source_image_encoding, image)
                except:
                    continue
                if similarity > max_similarity:
                    max_similarity = similarity
            similarity_array[identifier] = max_similarity
        return f"Face recognized: {self.labels[np.array(similarity_array).argmax()]} with {max(similarity_array)} similarity"


if __name__=='__main__':
    """
    This executes only if the code is executed manually

    Execute it by doing:
        python -m api.utils.recognition.classifiers.VGGFACE 
    """
    # #Use GPU in order to speed up training
    # from tensorflow.python.framework.ops import disable_eager_execution
    # disable_eager_execution()

    def test_with_cam():
        DELTA_RECOGNIZE = 5
        counter = 0
        cap = cv2.VideoCapture(0)
        while True:
            counter += 1
            success, frame = cap.read()
            if counter % DELTA_RECOGNIZE == 0:
                counter = 0
                output = classifier.recognize(frame)
                print(output)

            cv2.imshow('Webcam',cv2.flip(frame, 1))
            cv2.waitKey(1)

    classifier = VGGFACE()
    classifier.train()
    test_with_cam()