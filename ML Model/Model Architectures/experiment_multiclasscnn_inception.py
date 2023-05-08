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
from keras.layers import Dropout
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D,  \
    Dropout, Dense, Input, concatenate,      \
    GlobalAveragePooling2D, AveragePooling2D,\
    Flatten
from keras.utils import to_categorical
from scipy import signal
from scipy.interpolate import splev, splrep
import random
import os
import folium
from pyts.image import GramianAngularField
import seaborn as sns
import tensorflow_addons as tfa
sns.set()
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
path = "inception"

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

def plot_graphs(history, string):
    plt.figure(figsize=(7, 3))
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.savefig(path+"/model_"+string)
    plt.show()

kernel_init = tf.keras.initializers.glorot_uniform()
bias_init = tf.keras.initializers.Constant(value=0.2)
def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):
    
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    
    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)

    pool_proj = tf.keras.layers.MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

    output = tf.keras.layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    
    return output
# fit and evaluate a model
def evaluate_model(X_train, Y_train, X_test, Y_test):

    verbose, epochs, batch_size = 1, 100, 16
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], Y_train.shape[1]
    
    new_X_train = []
    for i in range(X_train.shape[0]):
        gaf = GramianAngularField()
        # image = []
        # print(np.array(X_train[i].flatten().reshape(-1, 1)).shape)
        new_X_train.append(gaf.fit_transform(X_train[i].flatten().reshape(1,-1))[0].reshape(360, 360, 1))
        # print(np.array(new_X_train).shape)
        # image.append(gaf.fit_transform(X_train[i].flatten().reshape(1,-1)))
        # print(np.array(image[0][0]).shape)
        # new_X_train.append(np.transpose(np.array(image), (3,2,1,0)))
    new_X_train = np.array(new_X_train)

    new_X_test = []
    for i in range(X_test.shape[0]):
        gaf = GramianAngularField()
        new_X_test.append(gaf.fit_transform(X_test[i].flatten().reshape(1,-1))[0].reshape(360, 360, 1))
        # image = []
        # for j in range(6):
        #     image.append(gaf.fit_transform(X_test[i].T[j].reshape((1,-1)))[0])
        # new_X_test.append(np.transpose(np.array(image), (1,2,0)))
    new_X_test = np.array(new_X_test)

    X_train = new_X_train
    X_test = new_X_test

    verbose, epochs, batch_size = 1, 50, 16
    n_features1, n_features2, n_features3, n_outputs = X_train.shape[1], X_train.shape[2], X_train.shape[3], Y_train.shape[1]
    print(n_features1, n_features2, n_features3, n_outputs)
    input_layer = Input(shape=(n_features1, n_features2, n_features3))

    # model = tf.keras.Sequential([Conv2D(filters=12, kernel_size=3, activation='relu', input_shape=(n_features1, n_features2, n_features3), kernel_regularizer=tf.keras.regularizers.L1L2(0.0001)),
    # MaxPooling2D(pool_size=(2,2), strides=2),
    # Conv2D(filters=16, kernel_size=5, activation='relu', kernel_regularizer=tf.keras.regularizers.L1L2(0.0001)),
    # MaxPooling2D(pool_size=(2,2), strides=2),
    # # Flatten(),
    # tf.keras.layers.Reshape((87, 87*16), input_shape=(87,87,16)),
    
    # tf.keras.layers.Dropout(0.4),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8, activation='relu', kernel_regularizer=tf.keras.regularizers.L1L2(0.0001))),
    # tf.keras.layers.Dropout(0.2),
    # Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.L1L2(0.0001)), 
    # Dense(n_outputs, activation='sigmoid')])
    x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
    x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
    x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

    x = inception_module(x,
                        filters_1x1=64,
                        filters_3x3_reduce=96,
                        filters_3x3=128,
                        filters_5x5_reduce=16,
                        filters_5x5=32,
                        filters_pool_proj=32,
                        name='inception_3a')

    x = inception_module(x,
                        filters_1x1=128,
                        filters_3x3_reduce=128,
                        filters_3x3=192,
                        filters_5x5_reduce=32,
                        filters_5x5=96,
                        filters_pool_proj=64,
                        name='inception_3b')

    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)

    x = inception_module(x,
                        filters_1x1=192,
                        filters_3x3_reduce=96,
                        filters_3x3=208,
                        filters_5x5_reduce=16,
                        filters_5x5=48,
                        filters_pool_proj=64,
                        name='inception_4a')


    x1 = AveragePooling2D((5, 5), strides=3)(x)
    x1 = Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
    x1 = Flatten()(x1)
    x1 = Dense(1024, activation='relu')(x1)
    x1 = Dropout(0.7)(x1)
    x1 = Dense(4, activation='softmax', name='auxilliary_output_1')(x1)

    x = inception_module(x,
                        filters_1x1=160,
                        filters_3x3_reduce=112,
                        filters_3x3=224,
                        filters_5x5_reduce=24,
                        filters_5x5=64,
                        filters_pool_proj=64,
                        name='inception_4b')

    x = inception_module(x,
                        filters_1x1=128,
                        filters_3x3_reduce=128,
                        filters_3x3=256,
                        filters_5x5_reduce=24,
                        filters_5x5=64,
                        filters_pool_proj=64,
                        name='inception_4c')

    x = inception_module(x,
                        filters_1x1=112,
                        filters_3x3_reduce=144,
                        filters_3x3=288,
                        filters_5x5_reduce=32,
                        filters_5x5=64,
                        filters_pool_proj=64,
                        name='inception_4d')


    x2 = AveragePooling2D((5, 5), strides=3)(x)
    x2 = Conv2D(128, (1, 1), padding='same', activation='relu')(x2)
    x2 = Flatten()(x2)
    x2 = Dense(1024, activation='relu')(x2)
    x2 = Dropout(0.7)(x2)
    x2 = Dense(4, activation='softmax', name='auxilliary_output_2')(x2)

    x = inception_module(x,
                        filters_1x1=256,
                        filters_3x3_reduce=160,
                        filters_3x3=320,
                        filters_5x5_reduce=32,
                        filters_5x5=128,
                        filters_pool_proj=128,
                        name='inception_4e')

    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)

    x = inception_module(x,
                        filters_1x1=256,
                        filters_3x3_reduce=160,
                        filters_3x3=320,
                        filters_5x5_reduce=32,
                        filters_5x5=128,
                        filters_pool_proj=128,
                        name='inception_5a')

    x = inception_module(x,
                        filters_1x1=384,
                        filters_3x3_reduce=192,
                        filters_3x3=384,
                        filters_5x5_reduce=48,
                        filters_5x5=128,
                        filters_pool_proj=128,
                        name='inception_5b')

    x = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)

    x = Dropout(0.4)(x)

    x = Dense(4, activation='softmax', name='output')(x)

    model = Model(input_layer, [x, x1, x2], name='inception_v1')


    # model = tf.keras.Sequential([
    # Conv2D(filters=6, kernel_size=3, activation='relu', input_shape=(n_features1, n_features2, n_features3), kernel_regularizer=tf.keras.regularizers.L1L2(0.0001)),
    # Conv2D(filters=12, kernel_size=3, activation='relu', kernel_regularizer=tf.keras.regularizers.L1L2(0.0001)),
    # Conv2D(filters=6, kernel_size=5, activation='relu', kernel_regularizer=tf.keras.regularizers.L1L2(0.0001)),
    # Conv2D(filters=12, kernel_size=3, activation='relu', kernel_regularizer=tf.keras.regularizers.L1L2(0.0001)),
    # Conv2D(filters=24, kernel_size=1, activation='relu', kernel_regularizer=tf.keras.regularizers.L1L2(0.0001)),
    # Flatten(),
    # Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.L1L2(0.0001)),
    # Dense(12, activation='relu', kernel_regularizer=tf.keras.regularizers.L1L2(0.0001)),
    # Dense(n_outputs, activation='sigmoid')])

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    # fit network
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=True)

    plot_graphs(history, 'categorical_accuracy')
    plot_graphs(history, 'loss')

    model_prediction = model.predict(X_test)

    predict = []
    onehot_predict = []
    for i in model_prediction:
        index = np.argmax(i)
        predict.append(index)
        if index == 0:
            onehot_predict.append([1, 0, 0, 0])
        elif index == 1:
            onehot_predict.append([0, 1, 0, 0])
        elif index == 2:
            onehot_predict.append([0, 0, 1, 0])
        elif index == 3:
            onehot_predict.append([0, 0, 0, 1])

    pothole_locations = set()
    speedbreaker_locations = set()
    badroads_locations = set()
    normalroads_locations = set()
    for i in range(len(predict)):
        if predict[i] == 0:
            normalroads_locations.add(tuple(loc_test_data[i]))
        if predict[i] == 1:
            pothole_locations.add(tuple(loc_test_data[i]))
        if predict[i] == 2:
            badroads_locations.add(tuple(loc_test_data[i]))
        if predict[i] == 3:
            speedbreaker_locations.add(tuple(loc_test_data[i]))

    normalroads_locations = list(normalroads_locations)
    pothole_locations = list(pothole_locations)
    badroads_locations = list(badroads_locations)
    speedbreaker_locations = list(speedbreaker_locations)

    # evaluate model
    _, accuracy, precision, recall = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=0)

    f1 = tfa.metrics.F1Score(num_classes=4)
    f1.update_state(Y_test, model_prediction)
    f1 = f1.result()
    
    def show_confusion_matrix(cm, labels):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, xticklabels=labels, yticklabels=labels, 
                    annot=True, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.savefig(path+"/confusion_matrix")
        plt.show()

    test_encode = []
    for i in Y_test:
        index = np.argmax(i)
        test_encode.append(index)

    confusion_mtx = tf.math.confusion_matrix(test_encode, predict)
    show_confusion_matrix(confusion_mtx, ["Normal Road", "Pothole", "Bad Road", "Speedbreaker"])

    this_map = folium.Map(prefer_canvas=True)
    for i in pothole_locations:
        folium.CircleMarker(location=[i[0], i[1]],
                        radius=6,
                        weight=10, color="black").add_to(this_map)
    for i in badroads_locations:
        folium.CircleMarker(location=[i[0], i[1]],
                        radius=6,
                        weight=10, color="black").add_to(this_map)
    for i in speedbreaker_locations:
        folium.CircleMarker(location=[i[0], i[1]],
                        radius=6,
                        weight=10, color="black").add_to(this_map)
    for i in range(len(loc_test_data)):
        if Y_test[i][0] == 1:
            folium.CircleMarker(location=[loc_test_data[i][0], loc_test_data[i][1]],
                            radius=5,
                            weight=5, color="white", fill=True).add_to(this_map)
        elif Y_test[i][1] == 1:
            folium.CircleMarker(location=[loc_test_data[i][0], loc_test_data[i][1]],
                            radius=5,
                            weight=5, color="brown", fill=True).add_to(this_map)
        elif Y_test[i][2] == 1:
            folium.CircleMarker(location=[loc_test_data[i][0], loc_test_data[i][1]],
                            radius=5,
                            weight=5, color="yellow", fill=True).add_to(this_map)
        elif Y_test[i][3] == 1:
            folium.CircleMarker(location=[loc_test_data[i][0], loc_test_data[i][1]],
                            radius=5,
                            weight=5, color="grey", fill=True).add_to(this_map)

    #Set the zoom to the maximum possible
    this_map.fit_bounds(this_map.get_bounds())

    #Save the map to an HTML file
    this_map.save(path+'/folium_visualization.html')

    return accuracy, precision, recall, f1

# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


# repeat experiment
scores = list()
repeats = 1
for r in range(repeats):
    score, precision, recall, f1 = evaluate_model(X_train, Y_train, X_test, Y_test)
    score = score * 100.0
    print('>#%d: Accuracy->%.3f, Precision->%.3f, Recall->%.3f, F1->%.3f' % (r, score, precision, recall, f1[0]))
    scores.append(score)
    # summarize results
    summarize_results(scores)