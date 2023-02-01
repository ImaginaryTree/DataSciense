#!/usr/bin/env python
# coding: utf-8

# # Importing Library

# In[2]:


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

# In[3]:


def GrayCoMatrix(im):
    g = skimage.feature.greycomatrix(im, [1], [0], 256, symmetric=True, normed=True)
    a1 = skimage.feature.greycoprops(g, 'contrast')[0][0]
    a2 = skimage.feature.greycoprops(g, 'energy')[0][0]
    a3 = skimage.feature.greycoprops(g, 'homogeneity')[0][0]
    a4 = skimage.feature.greycoprops(g, 'correlation')[0][0]
    a5 = skimage.feature.greycoprops(g, 'dissimilarity')[0][0]
    a6 = skimage.feature.greycoprops(g, 'ASM')[0][0]
    a7 = skimage.measure.shannon_entropy(g)
    temp = [a1, a2, a3, a4, a5, a6, a7];
    return temp

def list2df(data):
    x_test = pd.DataFrame(data, columns=['contrast', 'energy','homogeneity', 'correlation', 'dissimilarity', 'ASM', 'Entrophy'])
    #x_test = pd.DataFrame(data, columns=['contrast'])
    x_test_normalized = Normalizer(x_test)
    clsdata = classifier(x_test_normalized)
    
    return clsdata

def Normalizer(x_train):
    for column in x_train:
        x_train[column] = (x_train[column] - x_train[column].mean()) / x_train[column].std()
    return x_train


def classifier(x_test):
    y_pred = ExtraTreesCls.predict(x_test)
    pred = y_pred[len(y_pred)-1]
    return pred
        
def rectangle(startPoint, endPoint, pred = 1):
    color = (255, 0, 0) if pred == 1 else (0, 255, 0)
    thickness = 2
    image = cv.rectangle(resize, startPoint, endPoint, color, thickness)
    return image


# # Read csv file & data Training

# In[4]:


data_train = pd.read_csv("tes/new/train5.csv")
X_train = data_train.drop(('class'), axis=1)
Y_train = data_train['class']
x_train_normalized = Normalizer(X_train)
#x_train_normalized.drop(['dissimilarity', 'energy', 'homogeneity','correlation','ASM','Entrophy'], axis=1, inplace=True)
#x_train = X_train.values.tolist()
#y_train = Y_train.values.tolist()
ExtraTrees = ExtraTreesClassifier(max_depth=30, n_estimators=320, random_state=4)
ExtraTreesCls = ExtraTrees.fit(x_train_normalized, Y_train)


# # Running Camera with Opencv

# In[15]:


cam = cv.VideoCapture('Video/vid1.mp4')
xx = 0
yy = 240
left = []
middle = []
right = []
analysis = []
number=0
while True:
    isTrue, frame = cam.read()
    if isTrue == False:
        break
    resize = cv.resize(frame, (640,480))
    gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)
    #print(gray)
    im = gray[yy:yy + 240, xx:xx + 213]
    im2 = gray[yy:yy + 240, xx + 213:xx + 426]
    im3 = gray[yy:yy + 240, xx + 426:xx + 640]
    temp = [GrayCoMatrix(im), GrayCoMatrix(im2), GrayCoMatrix(im3)]
    #converting and classification the data test
    left.append(temp[0])
    middle.append(temp[1])
    right.append(temp[2])
    if len(left) == 5:
        #leftData = list2df(left)
        #middleData = list2df(middle)
        #rightData = list2df(right)
        ROI = [left,middle,right]
        viz = list2df(ROI[number])
        
        rect_size = [
            [(0,240), (214,240), (431,240)],
            [(213,480), (430,480), (640,480)]
        ]
        #draw = rectangle(viz, (0,240), (213,480))  
        #draw = rectangle(viz, (214,240), (430,480))
        #draw = rectangle(viz, (431,240), (640,480))
        draw = rectangle(rect_size[0][number], rect_size[1][number], viz)
        cv.imshow('Unfitured video', draw)
        if number == 2:
            number -= 3
        number+=1
        left.clear()
        middle.clear()
        right.clear()
        if cv.waitKey(15) == ord('q'):
            break
    analysis.append(temp[1])
    
    
#release the recording device
cam.release()
#destroy the windows
cv.destroyAllWindows()


# In[ ]:





# In[ ]:




