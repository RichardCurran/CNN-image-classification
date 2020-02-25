from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
from os import listdir
from os.path import isdir, join
import pandas as pd
from collections import Counter
from keras.models import load_model
import random

model_path = "C:/Users/Richard/Desktop/Diss_OUTPUTS/CNN_models/6_v3CNN_final_run_150epochs.model"
img_path = "C:/Users/Richard/Desktop/dissertation/test"
class_pickle = "C:/Users/Richard/Desktop/Diss_OUTPUTS/CNN_models/6_v3_aug_bin/Lbin_150epochs.pickle"

model = load_model(model_path)
LBin = pickle.loads(open(class_pickle, "rb").read())

test_img = np.load('np_data_1/test_img.npy')
test_labels = np.load('np_data_1/test_labels.npy')

test_img, test_labels = shuffle(test_img,test_labels, random_state = random.randint(0,999))


cat = list(np.where(test_labels =='cat'))
cow = list(np.where(test_labels =='cow'))
dog = list(np.where(test_labels =='dog'))
road = list(np.where(test_labels =='road'))
sheep = list(np.where(test_labels =='sheep'))


STOP_sign = cv2.imread("C:/Users/Richard/Desktop/dissertation/STOP.jpg")
STOP_sign = cv2.resize(STOP_sign, (150, 150))
GO_sign = cv2.imread("C:/Users/Richard/Desktop/dissertation/GO.jpg")
GO_sign = cv2.resize(GO_sign, (150, 150))

test_ims = [cat[0][40],cow[0][17],dog[0][40],road[0][11], sheep[0][101]]

for im in test_ims:
    img = test_img[im]
    prediction = model.predict(img.reshape(1,64,64,1))[0]
    #print(prediction)
    idx = np.argmax(prediction)
    pred_label = LBin.classes_[idx]
    real = test_labels[im]
    if pred_label == 'road':
        print("GO!")
        print('Predicted: {} with {:.2f}% confidence'.format(pred_label,prediction[idx] * 100))
        print('Actual: {}'.format(real))
        print("\n")
        cv2.imshow("go", GO_sign)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
    else:
        print("STOP!")
        print('Predicted: {} or not an empty road with {:.2f}% confidence'.format(pred_label,prediction[idx] * 100))
        print('Actual: {}'.format(real))
        print("\n")
        cv2.imshow("stop",STOP_sign)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
    cv2.imshow('{}'.format(real), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




