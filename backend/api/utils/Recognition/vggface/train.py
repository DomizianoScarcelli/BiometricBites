import os
import pickle

from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras_vggface.vggface import VGGFace

from bsproject.settings import MODELS_ROOT, LABELS_ROOT, SAMPLES_ROOT


def train_model():
    """
    Train the model with the new images
    """
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        SAMPLES_ROOT,
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
    model.save(os.path.join(MODELS_ROOT, 'transfer_learning_trained_face_cnn_model.h5'))

    class_dictionary = train_generator.class_indices
    class_dictionary = {
        value:key for key, value in class_dictionary.items()
    }
    # save the class dictionary to pickle
    face_label_filename = os.path.join(LABELS_ROOT, 'face-labels.pickle')
    if os.path.exists(face_label_filename):
        os.remove(face_label_filename)
    with open(face_label_filename, 'wb') as f: 
        pickle.dump(class_dictionary, f)

