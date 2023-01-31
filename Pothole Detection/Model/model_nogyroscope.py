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
import seaborn as sns
import folium

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
include = ['Pothole', 'Bad Road', 'Speedbreaker']

def resample(data, old_fs, new_fs=2):
    t = np.arange(len(data)) / old_fs
    spl = splrep(t, data)
    t1 = np.arange((len(data))*new_fs) / (old_fs*new_fs)
    return splev(t1, spl)

def train_test_split(data, longitude, latitude):

    data = [[[i[0][0],i[0][1],i[0][2]], i[1]] for i in data]
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
            # new = []
            new_fs = int(np.random.uniform(5, 20))
            _x = resample(accx, 10, new_fs)
            _x = resample(_x, new_fs, 1/new_fs)
            _y = resample(accy, 10, new_fs)
            _y = resample(_y, new_fs, 1/new_fs)
            _z = resample(accz, 10, new_fs)
            _z = resample(_z, new_fs, 1/new_fs)
            new = [[a,b,c] for a,b,c in zip(_x, _y, _z)]
            window.append([new, 'Pothole'])
            window_loc.append([potholes[index][1][0], potholes[index][1][1]])

        return window

    window = augment(window)

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

def plot_graphs(history, string):
    plt.figure(figsize=(7, 3))
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.savefig("no_gyroscope/model_"+string)
    plt.show()

# fit and evaluate a model
def evaluate_model(X_train, Y_train, X_test, Y_test):

    verbose, epochs, batch_size = 1, 200, 16
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], Y_train.shape[1]

    model = tf.keras.Sequential([Dense(32, activation='relu', input_shape=(n_timesteps,n_features)),
    Conv1D(filters=16, kernel_size=3, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.0005)),
    Conv1D(filters=32, kernel_size=3, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.0005)),
    MaxPooling1D(pool_size=2, strides=2),
    Conv1D(filters=32, kernel_size=5, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.0005)),
    Conv1D(filters=64, kernel_size=5, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.0005)),
    MaxPooling1D(pool_size=2, strides=2),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.0005))),
    tf.keras.layers.Dropout(0.2),
    Dense(n_outputs, activation='sigmoid')])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    # fit network
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=True)

    plot_graphs(history, 'binary_accuracy')
    plot_graphs(history, 'loss')

    predict = np.where(model.predict(X_test) > 0.5, 1, 0)
    pothole_locations = set()
    for i in range(len(predict)):
        if predict[i] == 1:
            pothole_locations.add(tuple(loc_test_data[i]))
    pothole_locations = list(pothole_locations)
    plt.scatter([y for x,y in pothole_locations], [x for x,y in pothole_locations], color="black",label="Model Prediction",  alpha=0.8, marker='x')
    c = []
    for _ in Y_test:
        if _ == 0:
            c.append("r")
        else:
            c.append("b")
    plt.scatter([y for x,y in loc_test_data], [x for x,y in loc_test_data], c=c, label="Blue=Pothole, Red=Normal", alpha=1.0, s=5)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis('equal')
    plt.tight_layout()
    plt.legend()
    plt.savefig("no_gyroscope/matplotlib_visualization")
    plt.show()

    # print(pothole_locations)
    # evaluate model
    _, accuracy, precision, recall = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=0)
    
    def show_confusion_matrix(cm, labels):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, xticklabels=labels, yticklabels=labels, 
                    annot=True, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.savefig("no_gyroscope/confusion_matrix")
        plt.show()

    confusion_mtx = tf.math.confusion_matrix([i[0] for i in Y_test],predict)
    show_confusion_matrix(confusion_mtx, ["Pothole", "Normal Road"])

    this_map = folium.Map(prefer_canvas=True)
    for i in pothole_locations:
        folium.CircleMarker(location=[i[0], i[1]],
                        radius=6,
                        weight=10, color="black").add_to(this_map)
    for i in range(len(loc_test_data)):
        if Y_test[i] == 0:
            folium.CircleMarker(location=[loc_test_data[i][0], loc_test_data[i][1]],
                            radius=4,
                            weight=5, color="red", fill=True).add_to(this_map)
        else:
            folium.CircleMarker(location=[loc_test_data[i][0], loc_test_data[i][1]],
                            radius=4,
                            weight=5, color="blue", fill=True).add_to(this_map)

    #Set the zoom to the maximum possible
    this_map.fit_bounds(this_map.get_bounds())

    #Save the map to an HTML file
    this_map.save('no_gyroscope/folium_visualization.html')

    return accuracy, precision, recall

# summarize scores
# def summarize_results(scores):
#     print(scores)
#     m, s = np.mean(scores), np.std(scores)
#     print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


# repeat experiment
scores = list()
repeats = 1
for r in range(repeats):
    score, precision, recall = evaluate_model(X_train, Y_train, X_test, Y_test)
    score = score * 100.0
    print('>#%d: Accuracy->%.3f, Precision->%.3f, Recall->%.3f' % (r, score, precision, recall))
    # scores.append(score)
    # # summarize results
    # summarize_results(scores)


