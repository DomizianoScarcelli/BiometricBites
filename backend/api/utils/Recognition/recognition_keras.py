import numpy as np
import os
import cv2
import pickle
from keras_preprocessing.image import img_to_array
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.models import load_model

# returns a compiled model identical to the previous one
model = load_model(
    "/Users/dov/Library/Mobile Documents/com~apple~CloudDocs/dovsync/Documenti Universita/Biometric Systems/Project/Repos.nosync/BS-Project/backend/api/utils/Train/recognizers/transfer_learning_trained_face_cnn_model.h5")

# dimension of images
image_width = 224
image_height = 224
BASE_DIR = os.path.dirname(os.path.dirname( __file__ ))

# load the training labels
face_label_filename = BASE_DIR + '/Train/pickles/' + 'face-labels.pickle'
with open(face_label_filename, "rb") as f: 
    class_dictionary = pickle.load(f)

class_list = [value for _, value in class_dictionary.items()]
print(class_list)


# Video parameters
cap = cv2.VideoCapture(0)

facecascade = cv2.CascadeClassifier(BASE_DIR + '/cascades/data/haarcascade_frontalface_default.xml')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    image_array = np.array(frame, "uint8")

    for (x_, y_, w, h) in faces:
        # draw the face detected
        face_detect = cv2.rectangle(frame, (x_, y_), (x_+w, y_+h), (255, 0, 255), 2)

        # resize the detected face to 224x224
        size = (image_width, image_height)
        roi = image_array[y_: y_ + h, x_: x_ + w]
        resized_image = cv2.resize(roi, size)

        # prepare the image for prediction
        x = img_to_array(resized_image)
        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_input(x, version=1)

        # making prediction
        predicted_prob = model.predict(x)
        print(predicted_prob)
        predicted_label = class_list[predicted_prob[0].argmax()]
        print("Predicted face: " + predicted_label)
        print("============================\n")
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        stroke = 2
        cv2.putText(frame, predicted_label, (x_,y_), font, 1, color, stroke, cv2.LINE_AA)


    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit if..
    if cv2.waitKey(1) & 0xFF == 27: # ...'ESC' pressed
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

