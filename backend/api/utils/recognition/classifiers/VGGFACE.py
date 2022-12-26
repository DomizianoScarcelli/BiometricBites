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
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras_vggface.vggface import VGGFace

class VGGFACE(Classifier):
    def __init__(self) -> None:
        super().__init__()
        self.side_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        self.model = load_model(os.path.join(self.models_root, 'vggface_model.h5'))
        self.image_width = 224
        self.image_height = 224
        self.pickle_file_name = "face_labels_vggface.pickle"
        self.labels = self.load_labels()

        
    def load_labels(self):
        label_path = os.path.join(self.labels_root, self.pickle_file_name)
        if not os.path.exists(label_path):
            class_dictionary = {}
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

        
    def train(self):
        """
        Train the model with the new images
        """
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        train_generator = train_datagen.flow_from_directory(
            self.image_dir,
            target_size=(224,224),
            color_mode='rgb',
            batch_size=32,
            class_mode='categorical',
            shuffle=True)

        train_generator.class_indices.values()
        NO_CLASSES = len(train_generator.class_indices.values())

        model = VGGFace(include_top=False,
        model='vgg16',
        input_shape=(224, 224, 3))

        x = model.output

        x = GlobalAveragePooling2D()(x)

        x = Dense(1024, activation='relu')(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)

        # final layer with softmax activation
        preds = Dense(NO_CLASSES, activation='softmax')(x)

        model = Model(inputs=model.inputs, outputs=preds)
        model.summary()

        # don't train the first 19 layers - 0..18
        for layer in model.layers[:19]:
            layer.trainable = False

        # train the rest of the layers - 19 onwards
        for layer in model.layers[19:]:
            layer.trainable = True

        model.compile(optimizer='Adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        model.fit(train_generator,
        batch_size = 1,
        verbose = 1,
        epochs = 20)


        # creates a HDF5 file
        model.save(os.path.join(self.models_root, 'vggface_model.h5'))

        class_dictionary = train_generator.class_indices
        class_dictionary = {
            value:key for key, value in class_dictionary.items()
        }
        # save the class dictionary to pickle
        face_label_filename = os.path.join(self.labels_root, self.pickle_file_name)
        if os.path.exists(face_label_filename):
            os.remove(face_label_filename)
        with open(face_label_filename, 'wb') as f: 
            pickle.dump(class_dictionary, f)

    def recognize(self, frame):
        model = self.model
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        image_array = np.array(frame, "uint8")

        for (x_, y_, w, h) in faces:
            # draw the face detected
            cv2.rectangle(frame, (x_, y_), (x_+w, y_+h), (255, 0, 0), 2)

            # resize the detected face to 224x224
            size = (self.image_width, self.image_height)
            roi = image_array[y_: y_ + h, x_: x_ + w]
            resized_image = cv2.resize(roi, size)

            # prepare the image for prediction
            x = img_to_array(resized_image)
            x = np.expand_dims(x, axis=0)
            x = utils.preprocess_input(x, version=1)

            # making prediction
            predicted_prob = model.predict(x)
            print(predicted_prob)

            if predicted_prob[0][0] >= .8:
                predicted_label = self.labels[predicted_prob[0].argmax()]
                print("Predicted face: " + str(predicted_label))
                print("============================\n")

                super().draw_label(frame, str(predicted_label), x_, y_)
            else:
                predicted_label = "Unknown"
                print("Predicted face: " + predicted_label)
                print("============================\n")
                super().draw_label(frame, predicted_label, x_, y_)

        return frame