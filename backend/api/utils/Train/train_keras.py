import os
import pandas as pd
import numpy as np
import keras
import pickle
import matplotlib.pyplot as plt

from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras_vggface.vggface import VGGFace
from keras.models import load_model

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

BASE_DIR = os.path.dirname(os.path.dirname( __file__ ))

train_generator = train_datagen.flow_from_directory(
    '/Users/dov/Library/Mobile Documents/com~apple~CloudDocs/dovsync/Documenti Universita/Biometric Systems/Project/Repos.nosync/BS-Project/backend/samples',
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
model.save(
    BASE_DIR + "/Train/recognizers/"+
    'transfer_learning_trained' +
    '_face_cnn_model.h5')

class_dictionary = train_generator.class_indices
class_dictionary = {
    value:key for key, value in class_dictionary.items()
}
# save the class dictionary to pickle
face_label_filename = BASE_DIR + '/Train/pickles/' + 'face-labels.pickle'
if os.path.exists(face_label_filename):
    os.remove(face_label_filename)
with open(face_label_filename, 'wb') as f: 
    pickle.dump(class_dictionary, f)