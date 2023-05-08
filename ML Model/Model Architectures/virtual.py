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
from sklearn.preprocessing import minmax_scale

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
path = "virtual"
isExist = os.path.exists(path+"_GAF_Pothole")

if not isExist:
   os.makedirs(path+"_GAF_Pothole")
   os.makedirs(path+"_GAF_NormalRoad")
   os.makedirs(path+"_GAF_BadRoad")
   os.makedirs(path+"_GAF_Speedbreaker")
   os.makedirs(path+"_MTF_Pothole")
   os.makedirs(path+"_MTF_NormalRoad")
   os.makedirs(path+"_MTF_BadRoad")
   os.makedirs(path+"_MTF_Speedbreaker")
   os.makedirs(path+"_RF_Pothole")
   os.makedirs(path+"_RF_NormalRoad")
   os.makedirs(path+"_RF_BadRoad")
   os.makedirs(path+"_RF_Speedbreaker")
else:
    paths = [path+"_GAF_Pothole", path+"_GAF_NormalRoad", path+"_GAF_BadRoad", path+"_GAF_Speedbreaker", 
             path+"_MTF_Pothole", path+"_MTF_NormalRoad", path+"_MTF_BadRoad", path+"_MTF_Speedbreaker", 
             path+"_RF_Pothole", path+"_RF_NormalRoad", path+"_RF_BadRoad", path+"_RF_Speedbreaker", ]
    for i in range(len(paths)):
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
    p_count = 0
    b_count = 0
    s_count = 0
    n_count = 0

    assert len(data) > 2*window_size + 1
    count = 0
    for i in range(0, len(data)-window_size, stride):
        
        temp = data[i:i+window_size]
        without_labels = [i[0] for i in temp]
        potholes, badroads, normalroads, speedbreakers = 0, 0, 0, 0
        for j in temp:
            if j[1] == "Pothole":
                potholes += 1
            elif j[1] == "Bad Road":
                badroads += 1
            elif j[1] == "Normal Road":
                normalroads += 1
            elif j[1] == "Speedbreaker":
                speedbreakers += 1
        dic = {"potholes" : potholes, "bad roads": badroads, "normal roads": normalroads, "speedbreakers": speedbreakers}
      

        if dic["potholes"] >= 5:
            window.append([without_labels, 'Pothole'])
            p_count+=1
        elif dic['speedbreakers'] >= 5:
            window.append([without_labels, 'Speedbreakers'])
            s_count += 1
        elif dic['bad roads'] >= 5:
            window.append([without_labels, 'Bad road'])
            # print(dic)
            b_count += 1
        elif dic['normal roads'] >= 1:
            window.append([without_labels, 'Normal road'])
            n_count += 1
        else:
            continue

        window_loc.append([np.mean([latitude[j] for j in range(i, i+window_size)]), np.mean([longitude[j] for j in range(i, i+window_size)])])

    def augment(window, window_loc):
        potholes = []
        normals = []
        bads = []
        speedbreakers = []
        new_window = []
        new_window_loc = []
        n = 200


        for i in range(len(window)):
            if window[i][1] == "Pothole":
                potholes.append([window[i][0], window_loc[i]])
                new_window.append(window[i])
                new_window_loc.append(window_loc[i])

            elif window[i][1] == "Normal road":
                normals.append([window[i][0], window_loc[i]])

            elif window[i][1] == "Bad road":
                bads.append([window[i][0], window_loc[i]])

            elif window[i][1] == "Speedbreakers":
                speedbreakers.append([window[i][0], window_loc[i]])
                new_window.append(window[i])
                new_window_loc.append(window_loc[i])

        p = n-len(potholes)
        s = n-len(speedbreakers)

        for i in range(abs(p)):
            index = int(np.random.random()*len(potholes))
            temp = potholes[index][0]
            accx = [j[0] for j in temp]
            accy = [j[1] for j in temp]
            accz = [j[2] for j in temp]
            gyx = [j[3] for j in temp]
            gyy = [j[4] for j in temp]
            gyz = [j[5] for j in temp]

            new_fs = int(np.random.uniform(3, 6))
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
            new_window.append([new, 'Pothole'])
            new_window_loc.append([potholes[index][1][0], potholes[index][1][1]])

        for i in range(abs(s)):
            index = int(np.random.random()*len(speedbreakers))
            temp = speedbreakers[index][0]
            accx = [j[0] for j in temp]
            accy = [j[1] for j in temp]
            accz = [j[2] for j in temp]
            gyx = [j[3] for j in temp]
            gyy = [j[4] for j in temp]
            gyz = [j[5] for j in temp]

            new_fs = int(np.random.uniform(3, 6))
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
            new_window.append([new, 'Speedbreakers'])
            new_window_loc.append([speedbreakers[index][1][0], speedbreakers[index][1][1]])

        for i in range(n):
            index = int(np.random.random()*len(normals))
            new_window.append([normals[i][0], 'Normal road'])
            new_window_loc.append(normals[i][1])
        for i in range(n):
            index = int(np.random.random()*len(bads))
            new_window.append([bads[i][0], 'Bad road'])
            new_window_loc.append(bads[i][1])

        return new_window, new_window_loc


    print((p_count, b_count, s_count, n_count))
    # max_count = max(p_count, b_count, s_count, n_count)
    # if p_count < max_count:
    #     window = augment(window)
    window, window_loc = augment(window, window_loc)

    data = np.array(window, dtype=object)
    locs = np.array(window_loc, dtype=object)

    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    data, locs = unison_shuffled_copies(data, locs)

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
    if l == 'Normal road':
        training_labels.append([1, 0, 0, 0])
    elif l == 'Pothole':
        training_labels.append([0, 1, 0, 0])
    elif l == 'Bad road':
        training_labels.append([0, 0, 1, 0])
    elif l == 'Speedbreakers':
        training_labels.append([0, 0, 0, 1])

# Loop over all test examples
for s, l in test_data:
    testing_sequences.append(np.array(s))
    if l == 'Normal road':
        testing_labels.append([1, 0, 0, 0])
    elif l == 'Pothole':
        testing_labels.append([0, 1, 0, 0])
    elif l == 'Bad road':
        testing_labels.append([0, 0, 1, 0])
    elif l == 'Speedbreakers':
        testing_labels.append([0, 0, 0, 1])

# Convert labels lists to numpy array
X_train = np.array(training_sequences)
X_test = np.array(testing_sequences)
Y_train = np.array(training_labels).reshape(-1, 4)
Y_test = np.array(testing_labels).reshape(-1, 4)
# print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], Y_train.shape[1]

# fit and evaluate a model
def GAF(X_train, Y_train, X_test, Y_test):

    new_X_train = []
    trueflag = False
    falseflag = False
    for i in range(X_train.shape[0]):
        gaf = GramianAngularField()
        image = gaf.fit_transform(X_train[i].flatten().reshape(1,-1))[0].reshape(360, 360)
        if not trueflag and (Y_train[i] == [0,1,0,0]).all():
            plt.imshow(image)
            plt.savefig(path+"_GAF_Pothole/pothole_pic"+str(i))
        if not falseflag and (Y_train[i] == [1,0,0,0]).all():
            plt.imshow(image)
            plt.savefig(path+"_GAF_NormalRoad/normalroad_pic"+str(i))
        if not trueflag and (Y_train[i] == [0,0,1,0]).all():
            plt.imshow(image)
            plt.savefig(path+"_GAF_BadRoad/badroad_pic"+str(i))
        if not falseflag and (Y_train[i] == [0,0,0,1]).all():
            plt.imshow(image)
            plt.savefig(path+"_GAF_Speedbreaker/speedbreaker_pic"+str(i))

    # new_X_test = []
    # for i in range(X_test.shape[0]):
    #     gaf = GramianAngularField()
    #     image = []
    #     for j in range(6):
    #         image.append(gaf.fit_transform(X_test[i].T[j].reshape((1,-1)))[0])
    #     new_X_test.append(np.transpose(np.array(image), (1,2,0)))
    # new_X_test = np.array(new_X_test)

def MTF(X_train, Y_train, X_test, Y_test):

    trueflag = False
    falseflag = False
    new_X_train = []
    for i in range(X_train.shape[0]):
        gaf = GramianAngularField(sample_range=(-1, 1), method='d')
        # image = []
        # print(np.array(X_train[i].flatten().reshape(-1, 1)).shape)
        new_X_train.append(gaf.fit_transform(minmax_scale(X_train[i].flatten().reshape(1,-1), feature_range=(-1,1)))[0].reshape(360, 360, 1))
        # print(np.array(new_X_train).shape)
        # image.append(gaf.fit_transform(X_train[i].flatten().reshape(1,-1)))
        # print(np.array(image[0][0]).shape)
        # new_X_train.append(np.transpose(np.array(image), (3,2,1,0)))
    new_X_train = np.array(new_X_train)

    new_X_test = []
    for i in range(X_test.shape[0]):
        gaf = GramianAngularField(sample_range=(-1, 1), method='d')
        new_X_test.append(gaf.fit_transform(minmax_scale(X_test[i].flatten().reshape(1,-1), feature_range=(-1,1)))[0].reshape(360, 360, 1))
        # image = []
        # for j in range(6):
        #     image.append(gaf.fit_transform(X_test[i].T[j].reshape((1,-1)))[0])
        # new_X_test.append(np.transpose(np.array(image), (1,2,0)))
    new_X_test = np.array(new_X_test)
    # for i in range(X_train.shape[0]):
    #     gaf = MarkovTransitionField()
    #     image = []
    #     for j in range(6):
    #         image.append(gaf.fit_transform(X_train[i].T[j].reshape((1,-1)))[0])
    #     if not trueflag and Y_train[i][0] == 1:
    #         plt.imshow(image[5])
    #         plt.savefig(path+"_MTF_True/pothole_pic"+str(i))
    #         trueflag = True
    #     if not falseflag and Y_train[i][0] == 0:
    #         plt.imshow(image[5])
    #         plt.savefig(path+"_MTF_False/normalroad_pic"+str(i))
    #         falseflag = True
    #     new_X_train.append(np.transpose(np.array(image), (1,2,0)))
    # new_X_train = np.array(new_X_train)

    # new_X_test = []
    # for i in range(X_test.shape[0]):
    #     gaf = GramianAngularField()
    #     image = []
    #     for j in range(6):
    #         image.append(gaf.fit_transform(X_test[i].T[j].reshape((1,-1)))[0])
    #     new_X_test.append(np.transpose(np.array(image), (1,2,0)))
    # new_X_test = np.array(new_X_test)

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
# MTF(X_train, Y_train, X_test, Y_test)
# RF(X_train, Y_train, X_test, Y_test)