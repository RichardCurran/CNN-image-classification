
import matplotlib
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
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
from collections import Counter

##Image processing

print("----- IMAGE PROCESSING: PLEASE STANDBY------")

path = 'C:/Users/Richard/Desktop/dissertation/images'
np_data_path = 'C:/Users/Richard/Documents/Py_Projects/thesis2.0/image_processing'

folders = set(listdir(path))
folders = list(folders)
image_labels = {folders[i]: i for i in range(0, len(folders))}

# # creates a dictionary of image folder names and assigns a label to each folder
image_labels = {folders[i]: i for i in range(0, len(folders))}

## ------------------MAIN IMAGE PROCESSING------------------------------------------------------
class img_proc_unbalanced:
    # WIDTH = 64
    # HEIGHT = 64
    num_classes = len(listdir(path))
    def img_proc(WIDTH, HEIGHT):
        images = []
        labels =[]
        clss = []
        for folder in listdir(path):
            for file in listdir(path + '/' + folder):
                if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                    img = path + '/' + folder + '/' + file
                    img = cv2.imread(img)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.resize(img, (WIDTH, HEIGHT))
                    images.append(img)
                    labels.append(image_labels[folder])
                    clss.append(folder)

        images = np.array(images, dtype= 'float32') / 255.0
        labels = np.array(labels, dtype= 'int32')

        return images, labels, clss

images,labels, clss = img_proc_unbalanced.img_proc(WIDTH = 64, HEIGHT=64)

np.save(os.path.join(np_data_path + '/np_data_64', 'images'), images)
np.save(os.path.join(np_data_path+ '/np_data_64', 'labels'), labels)
np.save(os.path.join(np_data_path + '/np_data_64', 'clss'), clss)



