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
from unlabelled_data import get_data
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
import tensorflow_addons as tfa
from keras.utils import to_categorical
from scipy import signal
from scipy.interpolate import splev, splrep
import random
import os
import seaborn as sns
sns.set()
import folium
import firebase_admin
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import itertools
from firebase_admin import credentials
from firebase_admin import db
from scipy.interpolate import CubicSpline

def get_locs():

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


    def train_test_split(data, longitude, latitude):

        window = []
        window_loc = []
        window_size = 60
        stride = 30

        assert len(data) > 2*window_size + 1
        
        for i in range(0, len(data)-window_size, stride):
            if latitude[i] > 0.1:
                temp = data[i:i+window_size]
                without_labels = [[i[0], i[1], i[2]] for i in temp]
                csx = CubicSpline(np.arange(60),np.array(without_labels)[:,0])
                csy = CubicSpline(np.arange(60),np.array(without_labels)[:,1])
                csz = CubicSpline(np.arange(60),np.array(without_labels)[:,2])
                newx = csx(np.arange(0, 60, 0.5))
                newy = csy(np.arange(0, 60, 0.5))
                newz = csz(np.arange(0, 60, 0.5))
                new = np.vstack((newx, newy, newz)).T
                window.append(new)
                window_loc.append([latitude[i+30], longitude[i+30]])

        data = np.array(window, dtype=object)
        locs = np.array(window_loc, dtype=object)

        return data, locs


    data, longitude, latitude = get_data()
    X, locs = train_test_split(data, longitude, latitude)

    print(X.shape)
    X = np.asarray(X).astype('float32')

    # It can be used to reconstruct the model identically.
    reconstructed_model = tf.keras.models.load_model("Raasta_Model")

    predict = reconstructed_model.predict(X)

    pothole_locations = set()
    speedbreaker_locations = set()
    normalroads_locations = set()
    for i in range(len(predict)):
        agmax = np.argmax(predict[i])
        if agmax == 0:
            normalroads_locations.add(tuple(locs[i]))
        if agmax == 1:
            pothole_locations.add(tuple(locs[i]))
        if agmax == 2:
            speedbreaker_locations.add(tuple(locs[i]))

    normalroads_locations = list(normalroads_locations)
    pothole_locations = list(pothole_locations)
    speedbreaker_locations = list(speedbreaker_locations)

    return normalroads_locations, pothole_locations, speedbreaker_locations

if __name__ == "__main__":
    get_locs()