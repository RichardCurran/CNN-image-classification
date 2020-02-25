## Information on dataset:
# balance, sample images, and augmentation example

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
from os import  listdir
from os.path import  isdir, join
import string
import pandas as pd
from collections import Counter
from keras.utils import to_categorical
from matplotlib import cm
from keras.preprocessing.image import ImageDataGenerator


def data_load(np_path):
    images = np.load('{}/images.npy'.format(np_path))
    labels = np.load('{}/labels.npy'.format(np_path))
    clss = np.load('{}/clss.npy'.format(np_path))

    return images, labels, clss
images, labels, clss = data_load(np_path= 'np_data_64')

lb = LabelBinarizer()
clss = np.array(clss)
labels_2 = lb.fit_transform(clss)
onehot_labels = labels_2

# one hot encoder labels and their class names in dataframe
def onehot_class():
    x = list(np.unique(labels, axis=0))
    y = list(lb.classes_)
    z = list(np.unique(onehot_labels, axis=0))
    label_class = {'Label':x,'Class':y}
    label_class = pd.DataFrame(label_class)
    print(label_class)
    return label_class

onehot_class()

# plot of the number of animal images in each class
def plot_animals():
    Class = Counter(clss).keys()
    animal_count = Counter(clss).values()

    img_count = {'Class': list(Class), 'Count': list(animal_count)}
    img_count = pd.DataFrame(img_count)
    data_plot = img_count.plot(kind='bar', x='Class', y='Count')

    #plt.show()
    return img_count,  data_plot

img_count, data_plot = plot_animals()
plt.show(data_plot)
print(img_count)

# image sample of each of class after grayscale
path = 'C:/Users/Richard/Desktop/dissertation/images'
folders = set(listdir(path))
folders = list(folders)
# # creates a dictionary of image folder names and assigns a label to each folder
image_labels = {folders[i]: i for i in range(0, len(folders))}
def img_sample():
    images = []
    labels = []
    clss = []
    files = []

    for folder in listdir(path):
        # clss.append(folder)
        for file in random.sample(listdir(path + '/' + folder),1):
            if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                img = path + '/' + folder + '/' + file
                files.append(files)
                img = cv2.imread(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #img = cv2.resize(img, (WIDTH, HEIGHT))
                images.append(img)
                labels.append(image_labels[folder])
                clss.append(folder)

    f, axarr = plt.subplots(2, 3)
    axarr[0, 0].imshow(images[0], cmap=plt.cm.gray)
    axarr[0, 0].set_title('{},{}'.format(labels[0], clss[0]))
    axarr[0, 1].imshow(images[1], cmap=plt.cm.gray)
    axarr[0, 1].set_title('{},{}'.format(labels[1], clss[1]))
    axarr[0, 2].imshow(images[1], cmap=plt.cm.gray)
    axarr[0, 2].set_title('{},{}'.format(labels[1], clss[1]))
    axarr[1, 0].imshow(images[2], cmap=plt.cm.gray)
    axarr[1, 0].set_title('{},{}'.format(labels[2], clss[2]))
    axarr[1, 1].imshow(images[3], cmap=plt.cm.gray)
    axarr[1, 1].set_title('{},{}'.format(labels[3], clss[3]))
    axarr[1, 2].imshow(images[4], cmap=plt.cm.gray)
    axarr[1, 2].set_title('{},{}'.format(labels[4], clss[4]))



    plt.show()
img_sample()

#augmentation example
augmented_data = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=25,
    height_shift_range=0.2,
    shear_range= 0.15,
    zoom_range=0.15)
aug_sample = 'C:/Users/Richard/Desktop/dissertation/aug_sample'

## folder with sample augmented image for write-up
img = 'C:/Users/Richard/Desktop/dissertation/test/dog/OIP-BEHBTmgkR6KZMSO0Z7qACAHaJI.jpeg'
img = cv2.imread(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = np.array(img, dtype='float32')
img  = img.reshape(1,img.shape[0], img.shape[1], 1)
i = 0
for image in augmented_data.flow(img, batch_size=1,
                          save_to_dir=aug_sample, save_prefix='dog_aug', save_format='jpeg'):

    i += 1
    if i > 5:
        break

