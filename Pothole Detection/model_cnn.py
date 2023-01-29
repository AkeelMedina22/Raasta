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
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from scipy import signal
from scipy.interpolate import splev, splrep
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
include = ['Pothole', 'Bad Road', 'Speedbreaker',]

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

    assert len(data) > 2*window_size + 1

    for i in range(0, len(data)-window_size, stride):
        temp = data[i:i+window_size]
        without_labels = [i[0] for i in temp]
        if temp[window_size//2][1] in include:
            window.append([without_labels, 'Pothole'])
        else:
            window.append([without_labels, 'Not Pothole'])
        window_loc.append([latitude[i], longitude[i]])

    def augment(window):
        pothole_count = 0
        normal_count = 0
        for i in window:
            if i[1] == "Pothole":
                pothole_count+=1
            else:
                normal_count+=1

        for i in range(normal_count-pothole_count):
            index = int(np.random.random()*pothole_count)
            temp = window[index][0]
            new = []
            new_fs = int(np.random.uniform(10, 20))
            for j in range(6):
                _ = resample(temp[j], 10, new_fs)
                new.append(resample(_, new_fs, 10/new_fs))

            new = [[new[0][k],new[1][k],new[2][k],new[3][k],new[4][k],new[5][k]] for k in range(len(new[0]))]
            window.append([new, 'Pothole'])

        return window

    window = augment(window)
    random.shuffle(window)

    data = np.array(window, dtype=object)

    train_ratio = 0.5
    sequence_len = data.shape[0]

    train_data = data[0:int(sequence_len*train_ratio)]
    test_data = data[int(sequence_len*train_ratio):]

    return train_data, test_data, window_loc


data, longitude, latitude = get_data()
train_data, test_data, location = train_test_split(data, longitude, latitude)
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

# print(X_train.shape[1], X_train.shape[2], Y_train.shape[1])
n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], Y_train.shape[1]


# fit and evaluate a model
def evaluate_model(X_train, Y_train, X_test, Y_test):
    verbose, epochs, batch_size = 0, 10, 8
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], Y_train.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    predict = np.where(model.predict(X_test) > 0.5, 1, 0)
    pothole_locations = set()
    for i in range(len(predict)):
        if predict[i] == 1:
            pothole_locations.add(tuple(location[i]))

    print(pothole_locations)
    # evaluate model
    _, accuracy = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=0)
    return accuracy

# # # summarize scores
def summarize_results(scores):
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


# repeat experiment
scores = list()
repeats = 5
for r in range(repeats):
    score = evaluate_model(X_train, Y_train, X_test, Y_test)
    score = score * 100.0
    print('>#%d: %.3f' % (1, score))
    scores.append(score)
    # summarize results
    summarize_results(scores)


