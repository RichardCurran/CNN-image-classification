from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
from os import  listdir
from os.path import  isdir, join
import pandas as pd

## the models used throughout implementation
#


# class CNN_model:
#
#     def CNN_build_v1(w, h, d, clss):
#         IP_SHAPE = (w, h, d)
#
#         model = Sequential()
#
#         model.add(Conv2D(64, (3, 3), input_shape=IP_SHAPE, activation='relu', strides=2,
#                          padding='same'))
#         model.add(MaxPooling2D(pool_size=(2,2)))
#         model.add(BatchNormalization())
#
#
#         model.add(Conv2D(32, (3, 3), activation='relu',strides=2, padding='same'))
#         model.add(BatchNormalization())
#         model.add(MaxPooling2D(pool_size=(2,2)))
#         model.add(Dropout(0.3))
#
#         model.add(Flatten())
#         model.add(Dense(288, activation='relu'))
#         # model.add(BatchNormalization())
#         model.add(Dropout(0.4))
#
#         model.add(Dense(clss, activation='softmax'))
#
#         return model
#     #1 hidden layer, dropout 40% after hidden layer
#     def CNN_build_v2(w, h, d, clss):
#         IP_SHAPE = (w, h, d)
#
#         model = Sequential()
#         model.add(Conv2D(64, (3, 3), input_shape=IP_SHAPE, activation='relu', strides=2,
#                          padding='same'))
#         model.add(BatchNormalization())
#         model.add(MaxPooling2D(pool_size=(2,2)))
#
#         model.add(Dropout(0.2))
#         model.add(Conv2D(32, (3, 3), activation='relu',strides=2, padding='same'))
#         model.add(BatchNormalization())
#         model.add(MaxPooling2D(pool_size=(2,2)))
#         model.add(Dropout(0.3))
#
#         model.add(Flatten())
#         model.add(Dense(512, activation='relu'))
#         model.add(BatchNormalization())
#         model.add(Dropout(0.4))
#         model.add(Dense(clss, activation='softmax'))
#
#         return model
#
#         # 1 hidden layer, dropout 40% after hidden layer

class CNN2:
    # def CNN_build_2(w, h, d, clss):
    #     IP_SHAPE = (w, h, d)
    #
    #     model = Sequential()
    #     model.add(Conv2D(64, (3, 3), input_shape=IP_SHAPE, activation='relu',
    #                      padding='same'))
    #     model.add(BatchNormalization())
    #     model.add(MaxPooling2D(pool_size=(2, 2)))
    #     model.add(Conv2D(32, (3, 3), activation='relu',padding='same'))
    #     model.add(MaxPooling2D(pool_size=(2, 2)))
    #     model.add(Dropout(0.3))
    #     model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    #     model.add(MaxPooling2D(pool_size=(2, 2)))
    #     model.add(Dropout(0.3))
    #
    #     model.add(Flatten())
    #     model.add(Dense(512, activation='relu'))
    #     model.add(Dropout(0.5))
    #     model.add(Dense(clss, activation='softmax'))
    #
    #     return model
    #
    # def CNN_build_3(w, h, d, clss):
    #     IP_SHAPE = (w, h, d)
    #
    #     model = Sequential()
    #
    #     model.add(Conv2D(16, (3, 3), input_shape=IP_SHAPE, activation='relu',
    #                      padding='same'))
    #     model.add(MaxPooling2D(pool_size=(2, 2)))
    #     model.add(BatchNormalization())
    #     model.add(Dropout(0.3))
    #
    #     model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    #     model.add(BatchNormalization())
    #     model.add(MaxPooling2D(pool_size=(2, 2)))
    #     model.add(Dropout(0.3))
    #
    #     model.add(Flatten())
    #     model.add(Dense(512, activation='relu'))
    #     model.add(BatchNormalization())
    #     model.add(Dropout(0.4))
    #
    #     model.add(Dense(clss, activation='softmax'))
    #
    #     return model
    #

    ##this is the best one yet
    # tune this model a little bit to reduce val_loss
    def CNN_build_V1(w,h,d,clss):

        IP_SHAPE = (w, h, d)

        model = Sequential()
        model.add(Conv2D(64, (3, 3), input_shape=(w, h, d), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(16, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(288))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(5))
        model.add(Activation('softmax'))

        return  model

    def CNN_build_V2(w, h, d, clss):

        IP_SHAPE = (w, h, d)
        model = Sequential()

        model.add(Conv2D(32, (3, 3), input_shape=(h, w, d), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(16, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(16, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(5))
        model.add(Activation('softmax'))

        return model

    def CNN_build_6(w,h,d,clss):

        IP_SHAPE = (w,h,d)

        model = Sequential()
        model.add(Conv2D(64, (3, 3), input_shape=IP_SHAPE), padding='same')
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.25)) # add more droput out

        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))
        #
        # model.add(Conv2D(64, (3, 3)))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))

        # the model so far outputs 3D feature maps (height, width, features)

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(clss))
        model.add(Activation('softmax'))

        return model

    # def CNN_build_6_v2(w, h, d, clss):
    #     IP_SHAPE = (w, h, d)
    #
    #     model = Sequential()
    #     model.add(Conv2D(64, (3, 3), input_shape=IP_SHAPE), padding='same')
    #     model.add(Activation('relu'))
    #     model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    #     model.add(Conv2D(32, (3, 3), padding='same'))
    #     model.add(Activation('relu'))
    #     model.add(MaxPooling2D(pool_size=(2, 2)))
    #     model.add(Dropout(0.25))  # add more droput out
    #
    #     model.add(Conv2D(32, (3, 3), padding='same'))
    #     model.add(Activation('relu'))
    #     model.add(MaxPooling2D(pool_size=(2, 2)))
    #     model.add(Dropout(0.3))
    #
    #     model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    #     model.add(Dense(512))
    #     model.add(Activation('relu'))
    #     model.add(Dropout(0.5))
    #     model.add(Dense(clss))
    #     model.add(Activation('softmax'))
    #
    #     return model

    def CNN_build_v3(w, h, d, clss):
        IP_SHAPE = (w, h, d)

        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=IP_SHAPE, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(96, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))


        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(clss))
        model.add(Activation('softmax'))

        return model













