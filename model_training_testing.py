# TRAINING, VALIDATING, TESTING

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
from keras.callbacks import LearningRateScheduler
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
from random import randint
import pickle
import cv2
import os
from os import  listdir
from os.path import  isdir, join
import pandas as pd
from collections import Counter
from keras import callbacks
from keras.optimizers import Adam

path = 'C:/Users/Richard/Desktop/dissertation/images'
clss_labls_path = "C:/Users/Richard/Desktop/Diss_OUTPUTS/CNN_models/Reruns/CNN_build_5Llower_LR.pickle"
np_data_path = 'C:/Users/Richard/Documents/Py_Projects/thesis2.0/image_processing'

# 64 x 64 images
images = np.load('np_data_64/images.npy')
labels = np.load('np_data_64/labels.npy')
clss = np.load('np_data_64/clss.npy')

print(len(clss))
print(len(clss))
print(len(images))
print(len(labels))

# parameters
WIDTH = 64
HEIGHT = 64
num_classes = len(listdir(path))
LBin = LabelBinarizer()
batchSize = 32
EPOCHS = 75
opt = Adam(lr = 0.001, decay= 0.001/EPOCHS)
random_shuffle = randint(0,9999)


images, clss = shuffle(images,clss, random_state = random_shuffle)

def reshaping_():

    train_img, test_img, train_label, test_label = train_test_split(images, clss,
                                                                    test_size=0.06,
                                                                    random_state=355)
    train_img, val_img, train_label, val_label = train_test_split(train_img, train_label,
                                                                  test_size=0.12,
                                                                  random_state=78)

    train_img = train_img.reshape(len(train_img), WIDTH, HEIGHT,1)
    test_img= test_img.reshape(len(test_img), WIDTH, HEIGHT, 1)
    val_img = val_img.reshape(len(val_img), WIDTH, HEIGHT, 1)
    train_label = LBin.fit_transform(train_label)
    val_label = LBin.transform(val_label)
    test_label_names = test_label
    test_label = LBin.transform(test_label)

    return train_img, train_label,val_img, val_label, test_img, test_label, test_label_names

train_img, train_label,val_img, val_label, test_img, test_label, test_label_names = reshaping_()

np.save(os.path.join(np_data_path + '/np_data_1', 'test_img'), test_img)
np.save(os.path.join(np_data_path+ '/np_data_1', 'test_labels'), test_label_names)


# print("train", train_img.shape)
# print("val", val_img.shape)
# print("test", test_img.shape)
# print(len(test_label_names))
# print(test_label.shape)
# print(LBin.classes_)
# print(np.unique(train_label))

# augmenting training data
augmented_data = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=25,
    height_shift_range=0.1,
    fill_mode='nearest',
    zoom_range=0.15)

### Importing the CNN model
from image_processing.CNN_models import CNN2
model = CNN2.CNN_build_4(w = WIDTH, h = HEIGHT, d = 1, clss = num_classes)

# Compiling the network
model.compile(optimizer = opt, loss='categorical_crossentropy',
               metrics=['accuracy'])

model.summary()

print("_____----____ MODEL TRAINING_____----_____")
history = model.fit_generator(augmented_data.flow(train_img, train_label, batch_size=32),
                              steps_per_epoch=len(train_img) // 32,
                              epochs=EPOCHS,
                              validation_data=(val_img, val_label),
                              verbose=1)

print("_____----____ SAVING MODEL_____----_____")
model_save = "C:/Users/Richard/Desktop/Diss_OUTPUTS/Reruns"
#model.save(model_save + 'CNN_build_5Llower_LR.model') # save each trained model to a folder to load in testing script


print("_____----____ MODEL VALIDATING_____----_____")
print('Average training accuracy: ', np.mean(history.history['accuracy'])*100) # average train accuracy
print('Average validation  accuracy:', np.mean(history.history['val_accuracy'])*100) # avg val. accuracy
print('Average training loss: ', np.mean(history.history['loss'])) # average train accuracy
print('Average validation loss:', np.mean(history.history['val_loss'])) # avg val. accuracy


# plot of training/val loss/accuracy
def train_val_plots(history):
    loss_vals = history['loss']
    val_loss_vals = history['val_loss']
    acc_vals = history['accuracy']
    val_acc_vals = history['val_accuracy']
    epochs = range(1, EPOCHS + 1)

    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))
    ax[0].plot(epochs, loss_vals, color='black', marker='o', linestyle=' ', label='Training Loss')
    ax[0].plot(epochs, val_loss_vals, color='red', marker='*', label='Validation Loss')
    ax[0].set_title('Training & Validation Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy/loss')
    ax[0].legend(loc='best')
    ax[0].grid(True)
    ax[1].plot(epochs, acc_vals, color='black', marker='o', linestyle=' ', label='Training Accuracy')
    ax[1].plot(epochs, val_acc_vals, color='red', marker='*', label='Validation Accuracy')
    ax[1].set_title('Training & Validation Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy/loss')
    ax[1].legend(loc='best')
    ax[1].grid(True)

    plt.show()

train_val_plots(history.history)

#classication report printed after each model is trained and validated
p = model.predict(test_img, batch_size= 1)
print(classification_report(test_label.argmax(axis = 1),
                            p.argmax(axis=1),
                            target_names= LBin.classes_))
