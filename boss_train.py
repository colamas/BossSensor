# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import random

import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras import layers

from boss_input import extract_data, resize_with_pad, IMAGE_SIZE


class Dataset(object):

    def __init__(self):
        self.X_train = None
        self.X_valid = None
        self.X_test = None
        self.Y_train = None
        self.Y_valid = None
        self.Y_test = None

    def read(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE, img_channels=3, nb_classes=2, path='./data/'):
        images, labels = extract_data(path)
        labels = np.reshape(labels, [-1])
        # 70% train, then split the remaining 30% evenly into valid and test
        X_train, X_rest, y_train, y_rest = train_test_split(images, labels, test_size=0.3, random_state=random.randint(0, 100))
        X_valid, X_test, y_valid, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=random.randint(0, 100))

        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_channels)
        X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, img_channels)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_channels)

        # the data, shuffled and split between train and test sets
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_valid.shape[0], 'valid samples')
        print(X_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        self.Y_train = keras.utils.to_categorical(y_train, nb_classes)
        self.Y_valid = keras.utils.to_categorical(y_valid, nb_classes)
        self.Y_test = keras.utils.to_categorical(y_test, nb_classes)

        self.X_train = X_train.astype('float32') / 255
        self.X_valid = X_valid.astype('float32') / 255
        self.X_test = X_test.astype('float32') / 255


class Model(object):

    FILE_PATH = './store/model.keras'

    def __init__(self):
        self.model = None

    def build_model(self, dataset, nb_classes=2, data_augmentation=True):
        self.model = keras.Sequential()
        self.model.add(keras.Input(shape=dataset.X_train.shape[1:]))

        if data_augmentation:
            # real-time data augmentation, only active during training
            self.model.add(layers.RandomRotation(20 / 360))
            self.model.add(layers.RandomTranslation(0.2, 0.2))
            self.model.add(layers.RandomFlip('horizontal'))

        self.model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Dropout(0.25))

        self.model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Dropout(0.25))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(512, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(nb_classes, activation='softmax'))

        self.model.summary()

    def train(self, dataset, batch_size=32, nb_epoch=40):
        # let's train the model using SGD + momentum (how original).
        sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])
        self.model.fit(dataset.X_train, dataset.Y_train,
                       batch_size=batch_size,
                       epochs=nb_epoch,
                       validation_data=(dataset.X_valid, dataset.Y_valid),
                       shuffle=True)

    def save(self, file_path=FILE_PATH):
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.model.save(file_path)
        print('Model Saved.')

    def load(self, file_path=FILE_PATH):
        self.model = keras.models.load_model(file_path)
        print('Model Loaded.')

    def predict(self, image):
        if image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_with_pad(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        image = image.astype('float32') / 255
        result = self.model.predict(image, verbose=0)
        print(result)

        return int(np.argmax(result[0]))

    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.X_test, dataset.Y_test, verbose=0)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))


if __name__ == '__main__':
    dataset = Dataset()
    dataset.read()

    model = Model()
    model.build_model(dataset)
    model.train(dataset, nb_epoch=10)
    model.save()

    model = Model()
    model.load()
    model.evaluate(dataset)
