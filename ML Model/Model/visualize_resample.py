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

# def resample(data, old_fs, new_fs=2):
#     t = np.arange(len(data)) / old_fs
#     spl = splrep(t, data)
#     t1 = np.arange((len(data))*new_fs) / (old_fs*new_fs)
#     return splev(t1, spl)


data, longitude, latitude = get_data()
y = [i[0][2] for i in data[0:200]]
x = np.arange(len(y))

fig, axs = plt.subplots(4)
axs[0].plot(x, y)
axs[1].plot(x, signal.resample(signal.resample(y, len(y)//4), len(y)))
axs[2].plot(x, signal.resample(signal.resample(y, len(y)*4), len(y)))
var = np.max(y)-np.min(y)
axs[3].plot(x, y+np.random.uniform(-1, 1, size=np.array(y).shape))
axs[0].title.set_text('Original')
axs[1].title.set_text('Undersampled')
axs[2].title.set_text('Oversampled')
axs[3].title.set_text('Synthetic')
plt.tight_layout()
plt.show()