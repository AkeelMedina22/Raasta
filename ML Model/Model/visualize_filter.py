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
from scipy import signal
import random
import os
import folium
from pyts.image import GramianAngularField
import seaborn as sns
import tensorflow_addons as tfa
sns.set()

def butter(data, fs=10, fc=2.5, order=2):
    # fc = frequency cutoff
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(order, w, 'highpass', analog=False)
    return signal.filtfilt(b, a, data)

def bessel(data, fs=10, fc=2.5, order=2):
    # fc = frequency cutoff
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.bessel(order, w, 'highpass', analog=False)
    return signal.filtfilt(b, a, data)

def chebyshev(data, fs=10, fc=2.5, order=2):
    # fc = frequency cutoff
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.cheby1(order, 7.5, w, 'highpass', analog=False)
    return signal.filtfilt(b, a, data)


data, longitude, latitude = get_data()
y = [i[0][2] for i in data[1200:1500]]
x = np.arange(len(y))

fig, axs = plt.subplots(4)
axs[0].plot(x, y)
axs[1].plot(x, butter(y))
axs[2].plot(x, bessel(y))
axs[3].plot(x, chebyshev(y))
axs[0].title.set_text('Original')
axs[1].title.set_text('High-pass Butterworth filter')
axs[2].title.set_text('High-pass Bessel filter')
axs[3].title.set_text('High-pass Chebyshev filter')

plt.tight_layout()
plt.show()