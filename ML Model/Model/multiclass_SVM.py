import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from data import get_data
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from scipy import signal
from scipy.interpolate import splev, splrep
import random
import os
import seaborn as sns
sns.set()
import folium
from sklearn.svm import SVC
from sklearn.tree import export_graphviz
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import tensorflow_addons as tfa

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
path = "multiclass_SVM/"
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
        n = 250


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

            if np.random.random() < 0.5:

                new_fs = int(np.random.uniform(2, 4))
                _x = signal.resample(signal.resample(accx, len(accx)//new_fs), len(accx))
                _y = signal.resample(signal.resample(accy, len(accy)//new_fs), len(accy))
                _z = signal.resample(signal.resample(accz, len(accz)//new_fs), len(accz))
                _gx = signal.resample(signal.resample(gyx, len(gyz)//new_fs), len(gyx))
                _gy = signal.resample(signal.resample(gyy, len(gyy)//new_fs), len(gyy))
                _gz = signal.resample(signal.resample(gyz, len(gyz)//new_fs), len(gyz))
                new = [[a,b,c,d,e,f] for a,b,c,d,e,f in zip(_x, _y, _z, _gx, _gy, _gz)]
                new_window.append([new, 'Pothole'])
                new_window_loc.append([potholes[index][1][0], potholes[index][1][1]])
            else:
                _x = accx+np.random.uniform(-2, 2, size=np.array(accx).shape)
                _y = accy+np.random.uniform(-2, 2, size=np.array(accx).shape)
                _z = accz+np.random.uniform(-2, 2, size=np.array(accx).shape)
                _gx = gyx+np.random.uniform(-2, 2, size=np.array(accx).shape)
                _gy = gyy+np.random.uniform(-2, 2, size=np.array(accx).shape)
                _gz = gyz+np.random.uniform(-2, 2, size=np.array(accx).shape)
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

            if np.random.random() < 0.5:
                new_fs = int(np.random.uniform(2, 4))
                _x = signal.resample(signal.resample(accx, len(accx)//new_fs), len(accx))
                _y = signal.resample(signal.resample(accy, len(accy)//new_fs), len(accy))
                _z = signal.resample(signal.resample(accz, len(accz)//new_fs), len(accz))
                _gx = signal.resample(signal.resample(gyx, len(gyz)//new_fs), len(gyx))
                _gy = signal.resample(signal.resample(gyy, len(gyy)//new_fs), len(gyy))
                _gz = signal.resample(signal.resample(gyz, len(gyz)//new_fs), len(gyz))
                new = [[a,b,c,d,e,f] for a,b,c,d,e,f in zip(_x, _y, _z, _gx, _gy, _gz)]
                new_window.append([new, 'Speedbreakers'])
                new_window_loc.append([speedbreakers[index][1][0], speedbreakers[index][1][1]])
            else:
                _x = accx+np.random.uniform(-2, 2, size=np.array(accx).shape)
                _y = accy+np.random.uniform(-2, 2, size=np.array(accx).shape)
                _z = accz+np.random.uniform(-2, 2, size=np.array(accx).shape)
                _gx = gyx+np.random.uniform(-2, 2, size=np.array(accx).shape)
                _gy = gyy+np.random.uniform(-2, 2, size=np.array(accx).shape)
                _gz = gyz+np.random.uniform(-2, 2, size=np.array(accx).shape)
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
        training_labels.append(0)
    elif l == 'Pothole':
        training_labels.append(1)
    elif l == 'Bad road':
        training_labels.append(2)
    elif l == 'Speedbreakers':
        training_labels.append(3)

# Loop over all test examples
for s, l in test_data:
    testing_sequences.append(np.array(s))
    if l == 'Normal road':
        testing_labels.append(0)
    elif l == 'Pothole':
        testing_labels.append(1)
    elif l == 'Bad road':
        testing_labels.append(2)
    elif l == 'Speedbreakers':
        testing_labels.append(3)

# Convert labels lists to numpy array
X_train = np.array(training_sequences)
X_test = np.array(testing_sequences)
Y_train = np.array(training_labels).reshape(-1, 1)
Y_test = np.array(testing_labels).reshape(-1, 1)
# print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], Y_train.shape[1]

def plot_graphs(history, string):
    plt.figure(figsize=(7, 3))
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.savefig(path+"model_"+string)
    plt.show()

# fit and evaluate a model
def evaluate_model(X_train, Y_train, X_test, Y_test):


    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
    X_train = preprocessing.StandardScaler().fit_transform(X_train)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
    X_test = preprocessing.StandardScaler().fit_transform(X_test)
    Y_train = np.ravel(Y_train)
    Y_test = np.ravel(Y_test)
    SVM = SVC(C=2.5, cache_size=6000, degree=6, tol=1e-9).fit(X_train, Y_train)
    print("training set score: %f" % SVM.score(X_train, Y_train))
    print("test set score: %f" % SVM.score(X_test, Y_test))

    m_predict = SVM.predict(X_test)

    print("f1 "+ str(f1_score(Y_test, m_predict, zero_division=1, average='macro')))
    print("recall " + str(recall_score(Y_test, m_predict, zero_division=1, average='macro')))
    print("precision " + str(precision_score(Y_test, m_predict, zero_division=1, average='macro')))


    # pothole_locations = set()
    # speedbreaker_locations = set()
    # badroads_locations = set()
    # normalroads_locations = set()
    # for i in range(len(m_predict)):
    #     if m_predict[i] == 0:
    #         normalroads_locations.add(tuple(loc_test_data[i]))
    #     if m_predict[i] == 1:
    #         pothole_locations.add(tuple(loc_test_data[i]))
    #     if m_predict[i] == 2:
    #         badroads_locations.add(tuple(loc_test_data[i]))
    #     if m_predict[i] == 3:
    #         speedbreaker_locations.add(tuple(loc_test_data[i]))

    # normalroads_locations = list(normalroads_locations)
    # pothole_locations = list(pothole_locations)
    # badroads_locations = list(badroads_locations)
    # speedbreaker_locations = list(speedbreaker_locations)


    # f1 = tfa.metrics.F1Score(num_classes=4)
    # f1.update_state(Y_test, m_predict)
    # f1 = f1.result()
    
    # def show_confusion_matrix(cm, labels):
    #     plt.figure(figsize=(10, 8))
    #     sns.heatmap(cm, xticklabels=labels, yticklabels=labels, 
    #                 annot=True, fmt='g')
    #     plt.xlabel('Prediction')
    #     plt.ylabel('Label')
    #     plt.savefig("main_results/confusion_matrix")
    #     plt.show()

    # test_encode = []
    # for i in Y_test:
    #     index = np.argmax(i)
    #     test_encode.append(index)

    # confusion_mtx = tf.math.confusion_matrix(test_encode, m_predict)
    # show_confusion_matrix(confusion_mtx, ["Normal Road", "Pothole", "Bad Road", "Speedbreaker"])

    # this_map = folium.Map(prefer_canvas=True)
    # for i in pothole_locations:
    #     folium.CircleMarker(location=[i[0], i[1]],
    #                     radius=6,
    #                     weight=10, color="black").add_to(this_map)
    # for i in badroads_locations:
    #     folium.CircleMarker(location=[i[0], i[1]],
    #                     radius=6,
    #                     weight=10, color="black").add_to(this_map)
    # for i in speedbreaker_locations:
    #     folium.CircleMarker(location=[i[0], i[1]],
    #                     radius=6,
    #                     weight=10, color="black").add_to(this_map)
    # for i in range(len(loc_test_data)):
    #     if Y_test[i][0] == 1:
    #         folium.CircleMarker(location=[loc_test_data[i][0], loc_test_data[i][1]],
    #                         radius=5,
    #                         weight=5, color="white", fill=True).add_to(this_map)
    #     elif Y_test[i][1] == 1:
    #         folium.CircleMarker(location=[loc_test_data[i][0], loc_test_data[i][1]],
    #                         radius=5,
    #                         weight=5, color="brown", fill=True).add_to(this_map)
    #     elif Y_test[i][2] == 1:
    #         folium.CircleMarker(location=[loc_test_data[i][0], loc_test_data[i][1]],
    #                         radius=5,
    #                         weight=5, color="yellow", fill=True).add_to(this_map)
    #     elif Y_test[i][3] == 1:
    #         folium.CircleMarker(location=[loc_test_data[i][0], loc_test_data[i][1]],
    #                         radius=5,
    #                         weight=5, color="grey", fill=True).add_to(this_map)

    # #Set the zoom to the maximum possible
    # this_map.fit_bounds(this_map.get_bounds())

    # #Save the map to an HTML file
    # this_map.save('main_results/folium_visualization.html')

    return 

evaluate_model(X_train, Y_train, X_test, Y_test)