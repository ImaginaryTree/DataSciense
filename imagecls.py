#!/usr/bin/env python
# coding: utf-8

# # Importing Library


import pandas as pd
import math
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import skimage.io
import skimage.feature
import matplotlib.pyplot as plt
#support algorithm for machine learning
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingRegressor


import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
import cv2 as cv


# # Normalization & classifier


def classifier(ExtraTreesCls,x_test):
    y_pred = ExtraTreesCls.predict(x_test)
    print(y_pred, end="")
    if y_pred[len(y_pred)-1] == 0:
        print("aspal -> ")
    elif y_pred[len(y_pred)-1] == 1:
        print("trotoar -> ")

def Normalizer(x_train):
    for column in x_train:
        x_train[column] = (x_train[column] - x_train[column].mean()) / x_train[column].std()
    return x_train


# # Read csv file & data Training


data_train = pd.read_csv("tes/new/train5.csv")
X_train = data_train.drop(('class'), axis=1)
Y_train = data_train['class']
x_train_normalized = Normalizer(X_train)
#x_train_normalized.drop(['dissimilarity', 'energy', 'homogeneity','correlation','ASM','Entrophy'], axis=1, inplace=True)
#x_train = X_train.values.tolist()
#y_train = Y_train.values.tolist()
#print(x_train_normalized)
ExtraTrees = ExtraTreesClassifier(max_depth=30, n_estimators=320, random_state=4)
ExtraTreesCls = ExtraTrees.fit(x_train_normalized, Y_train)
#cascade = cv.CascadeClassifier('cascade\haarcascade_fullbody.xml')


# # Running Camera with Opencv

cam = cv.VideoCapture('Video/trotoar.mp4')
xx = 184
yy = 1623
data = []
analysis = []
number=0
while True:
    isTrue, frame = cam.read()
    if isTrue == False:
        break
    resize = cv.resize(frame, (1080,1920))
    gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)
    #print(gray)
    im = gray[yy:yy + 297, xx:xx + 712]
    g = skimage.feature.greycomatrix(im, [1], [0], 256, symmetric=True, normed=True)
    a1 = skimage.feature.greycoprops(g, 'contrast')[0][0]
    a2 = skimage.feature.greycoprops(g, 'energy')[0][0]
    a3 = skimage.feature.greycoprops(g, 'homogeneity')[0][0]
    a4 = skimage.feature.greycoprops(g, 'correlation')[0][0]
    a5 = skimage.feature.greycoprops(g, 'dissimilarity')[0][0]
    a6 = skimage.feature.greycoprops(g, 'ASM')[0][0]
    a7 = skimage.measure.shannon_entropy(g)
    temp = [a1, a2, a3, a4, a5, a6, a7];
    #temp = [a1];
    data.append(temp)
    analysis.append(temp)
    if len(data) == 5:
        x_test = pd.DataFrame(data, columns=['contrast', 'energy','homogeneity', 'correlation', 'dissimilarity', 'ASM', 'Entrophy'])
        #x_test = pd.DataFrame(data, columns=['contrast'])
        x_test_normalized = Normalizer(x_test)
        classifier(ExtraTreesCls, x_test_normalized)
        data.clear()
    cv.imshow('Unfitured video', frame)
    if cv.waitKey(1) == ord('q'):
        break
#release the recording device
cam.release()
#destroy the window
cv.destroyAllWindows()





