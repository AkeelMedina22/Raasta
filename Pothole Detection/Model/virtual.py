# Paper: https://github.com/AkeelMedina22/Raasta/blob/main/Literature%20Review/A%20deep%20learning%20approach%20to%20automatic%20road%20surface%20monitoring%20and%20pothole%20detection%20-%20Important.pdf

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing
import matplotlib.pylab as plt
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from data import get_data
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
# cnn model
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import to_categorical
from scipy import signal
from scipy.interpolate import splev, splrep
import random
import os
import seaborn as sns
import folium
from pyts.image import GramianAngularField
from pyts.image import MarkovTransitionField
from pyts.image import RecurrencePlot


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
include = ['Pothole', 'Bad Road']
path = "virtual"
isExist = os.path.exists(path+"_GAF_True")

if not isExist:
   os.makedirs(path+"_GAF_True")
   os.makedirs(path+"_GAF_False")
   os.makedirs(path+"_MTF_True")
   os.makedirs(path+"_MTF_False")
   os.makedirs(path+"_RF_True")
   os.makedirs(path+"_RF_False")
else:
    paths = [path+"_GAF_True", path+"_GAF_False", path+"_MTF_True", path+"_MTF_False", path+"_RF_True", path+"_RF_False"]
    for i in range(6):
        for file_name in os.listdir(paths[i]):    
            file = path + file_name
            if os.path.isfile(file):
                os.remove(file)

def resample(data, old_fs, new_fs=2):
    t = np.arange(len(data)) / old_fs
    spl = splrep(t, data)
    t1 = np.arange((len(data))*new_fs) / (old_fs*new_fs)
    return splev(t1, spl)

def train_test_split(data, longitude, latitude):

    window = []
    window_loc = []
    window_size = 60
    stride = 30
    data_count = 0

    assert len(data) > 2*window_size + 1

    for i in range(0, len(data)-window_size, stride):
        temp = data[i:i+window_size]
        without_labels = [i[0] for i in temp]
        if any([1 if j[1] in include else 0 for j in temp]):
            window.append([without_labels, 'Pothole'])
            data_count+=1
        else:
            window.append([without_labels, 'Not Pothole'])
        window_loc.append([latitude[i], longitude[i]])

    def augment(window):
        potholes = []
        normals = []
        for i in range(len(window)):
            if window[i][1] == "Pothole":
                potholes.append([window[i][0], window_loc[i]])
            else:
                normals.append(window[i][0])
  
        for i in range(len(normals)-len(potholes)):
            index = int(np.random.random()*len(potholes))
            temp = potholes[index][0]
            accx = [j[0] for j in temp]
            accy = [j[1] for j in temp]
            accz = [j[2] for j in temp]
            gyx = [j[3] for j in temp]
            gyy = [j[4] for j in temp]
            gyz = [j[5] for j in temp]

            new_fs = int(np.random.uniform(6, 14))
            _x = resample(accx, 10, new_fs)
            _x = resample(_x, new_fs, 1/new_fs)
            _y = resample(accy, 10, new_fs)
            _y = resample(_y, new_fs, 1/new_fs)
            _z = resample(accz, 10, new_fs)
            _z = resample(_z, new_fs, 1/new_fs)
            _gx = resample(gyx, 10, new_fs)
            _gx = resample(_gx, new_fs, 1/new_fs)
            _gy = resample(gyy, 10, new_fs)
            _gy = resample(_gy, new_fs, 1/new_fs)
            _gz = resample(gyz, 10, new_fs)
            _gz = resample(_gz, new_fs, 1/new_fs)
            new = [[a,b,c,d,e,f] for a,b,c,d,e,f in zip(_x, _y, _z, _gx, _gy, _gz)]
            window.append([new, 'Pothole'])
            window_loc.append([potholes[index][1][0], potholes[index][1][1]])

        return window

    # if data_count/len(window) < 0.5:
    #     window = augment(window)

    data = np.array(window, dtype=object)
    locs = np.array(window_loc, dtype=object)

    # def unison_shuffled_copies(a, b):
    #     assert len(a) == len(b)
    #     p = np.random.permutation(len(a))
    #     return a[p], b[p]

    # data, locs = unison_shuffled_copies(data, locs)

    train_ratio = 0.8
    sequence_len = data.shape[0]

    train_data = data[0:int(sequence_len*train_ratio)]
    test_data = data[int(sequence_len*train_ratio):]

    loc_train_data = locs[0:int(sequence_len*train_ratio)]
    loc_test_data = locs[int(sequence_len*train_ratio):]

    return train_data, test_data, list(loc_train_data), list(loc_test_data)


data, longitude, latitude = get_data()
train_data, test_data, loc_train_data, loc_test_data = train_test_split(data, longitude, latitude)
# Initialize sequences and labels lists
training_sequences = []
training_labels = []

testing_sequences = []
testing_labels = []

# Loop over all training examples
for s, l in train_data:
    # print(np.array(s).shape, end = "\n\n\n")
    training_sequences.append(np.array(s))
    if l == 'Pothole':
        training_labels.append(1)
    else:
        training_labels.append(0)

# Loop over all test examples
for s, l in test_data:
    testing_sequences.append(np.array(s))
    if l == 'Pothole':
        testing_labels.append(1)
    else:
        testing_labels.append(0)

# Convert labels lists to numpy array
X_train = np.array(training_sequences)
X_test = np.array(testing_sequences)
Y_train = np.array(training_labels).reshape(-1, 1)
Y_test = np.array(testing_labels).reshape(-1, 1)

n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], Y_train.shape[1]

# fit and evaluate a model
def GAF(X_train, Y_train, X_test, Y_test):

    new_X_train = []
    trueflag = False
    falseflag = False
    for i in range(X_train.shape[0]):
        gaf = GramianAngularField()
        image = []
        for j in range(6):
            image.append(gaf.fit_transform(X_train[i].T[j].reshape((1,-1)))[0])
        if not trueflag and Y_train[i][0] == 1:
            plt.imshow(image[5])
            plt.savefig(path+"_GAF_True/pothole_pic"+str(i))
            trueflag = True
        if not falseflag and Y_train[i][0] == 0:
            plt.imshow(image[5])
            plt.savefig(path+"_GAF_False/normalroad_pic"+str(i))
            falseflag = True
        new_X_train.append(np.transpose(np.array(image), (1,2,0)))
    new_X_train = np.array(new_X_train)

    new_X_test = []
    for i in range(X_test.shape[0]):
        gaf = GramianAngularField()
        image = []
        for j in range(6):
            image.append(gaf.fit_transform(X_test[i].T[j].reshape((1,-1)))[0])
        new_X_test.append(np.transpose(np.array(image), (1,2,0)))
    new_X_test = np.array(new_X_test)

def MTF(X_train, Y_train, X_test, Y_test):

    new_X_train = []
    trueflag = False
    falseflag = False
    for i in range(X_train.shape[0]):
        gaf = MarkovTransitionField()
        image = []
        for j in range(6):
            image.append(gaf.fit_transform(X_train[i].T[j].reshape((1,-1)))[0])
        if not trueflag and Y_train[i][0] == 1:
            plt.imshow(image[5])
            plt.savefig(path+"_MTF_True/pothole_pic"+str(i))
            trueflag = True
        if not falseflag and Y_train[i][0] == 0:
            plt.imshow(image[5])
            plt.savefig(path+"_MTF_False/normalroad_pic"+str(i))
            falseflag = True
        new_X_train.append(np.transpose(np.array(image), (1,2,0)))
    new_X_train = np.array(new_X_train)

    new_X_test = []
    for i in range(X_test.shape[0]):
        gaf = GramianAngularField()
        image = []
        for j in range(6):
            image.append(gaf.fit_transform(X_test[i].T[j].reshape((1,-1)))[0])
        new_X_test.append(np.transpose(np.array(image), (1,2,0)))
    new_X_test = np.array(new_X_test)

def RF(X_train, Y_train, X_test, Y_test):

    new_X_train = []
    trueflag = False
    falseflag = False
    for i in range(X_train.shape[0]):
        gaf = RecurrencePlot()
        image = []
        for j in range(6):
            image.append(gaf.fit_transform(X_train[i].T[j].reshape((1,-1)))[0])
        if not trueflag and Y_train[i][0] == 1:
            plt.imshow(image[5])
            plt.savefig(path+"_RF_True/pothole_pic"+str(i))
            trueflag = True
        if not falseflag and Y_train[i][0] == 0:
            plt.imshow(image[5])
            plt.savefig(path+"_RF_False/normalroad_pic"+str(i))
            falseflag = True
        new_X_train.append(np.transpose(np.array(image), (1,2,0)))
    new_X_train = np.array(new_X_train)

    new_X_test = []
    for i in range(X_test.shape[0]):
        gaf = GramianAngularField()
        image = []
        for j in range(6):
            image.append(gaf.fit_transform(X_test[i].T[j].reshape((1,-1)))[0])
        new_X_test.append(np.transpose(np.array(image), (1,2,0)))
    new_X_test = np.array(new_X_test)

GAF(X_train, Y_train, X_test, Y_test)
MTF(X_train, Y_train, X_test, Y_test)
RF(X_train, Y_train, X_test, Y_test)