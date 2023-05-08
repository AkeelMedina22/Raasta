import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import folium
import firebase_admin
import numpy as np
from firebase_admin import credentials
from firebase_admin import db
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
from newdata import get_data
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
# import firebase_admin
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import itertools
# from firebase_admin import credentials
# from firebase_admin import db
from scipy import signal
import math
import json
from shapely.geometry import MultiPoint
from geopy.distance import great_circle

def get_data():

    latitude = []
    longitude = []
    labels = []
    id = []
    l50 = []
    la50 = []
    lo50 = []
    id50 = []


    with open('raasta-c542d-default-rtdb-sensor-data-export.json', 'r') as f:
    # with open('raasta-c542d-default-rtdb-sensor-data2-export.json', 'r') as f:

        data = json.load(f)
        for key in data:
            for timestamp in data[key]:
                for l in data[key][timestamp]:
                    if l == "latitude":
                        latitude.append(float(data[key][timestamp][l]))

                    elif l == "longitude":
                        longitude.append(float(data[key][timestamp][l]))

                    elif l == "label":
                        labels.append(data[key][timestamp][l])

                    elif l == "android-id":
                        id.append(data[key][timestamp][l])
    
    with open('raasta-c542d-default-rtdb-sensor-data2-export.json', 'r') as f:
    # with open('raasta-c542d-default-rtdb-sensor-data2-export.json', 'r') as f:

        data = json.load(f)
        for key in data:
            for timestamp in data[key]:
                for l in data[key][timestamp]:
                    if l == "latitude":
                        la50.append(float(data[key][timestamp][l]))

                    elif l == "longitude":
                        lo50.append(float(data[key][timestamp][l]))

                    elif l == "label":
                        l50.append(data[key][timestamp][l])
                    
                    elif l == "android-id":
                        id50.append(data[key][timestamp][l])


    return labels, longitude, latitude, l50, lo50, la50, id, id50

labels, longitude, latitude, l50, lo50, la50, id, id50 = get_data()

users = list(set(list(set(id)) + list(set(id50))))
this_map = folium.Map(prefer_canvas=True)

for i in range(len(latitude)):
    if id[i] == users[0]:
        folium.CircleMarker(location=[latitude[i], longitude[i]],
                            radius=2,
                            weight=2, color="black").add_to(this_map)
    if id[i] == users[1]:
        folium.CircleMarker(location=[latitude[i], longitude[i]],
                            radius=2,
                            weight=2, color="red").add_to(this_map)
    if id[i] == users[2]:
        folium.CircleMarker(location=[latitude[i], longitude[i]],
                            radius=2,
                            weight=2, color="blue").add_to(this_map)
    if id[i] == users[3]:
        folium.CircleMarker(location=[latitude[i], longitude[i]],
                            radius=2,
                            weight=2, color="green").add_to(this_map)
    if id[i] == users[4]:
        folium.CircleMarker(location=[latitude[i], longitude[i]],
                            radius=2,
                            weight=2, color="yellow").add_to(this_map)

for i in range(len(lo50)):
    if id50[i] == users[0]:
        folium.CircleMarker(location=[la50[i], lo50[i]],
                            radius=2,
                            weight=2, color="black").add_to(this_map)
    if id50[i] == users[1]:
        folium.CircleMarker(location=[la50[i], lo50[i]],
                            radius=2,
                            weight=2, color="red").add_to(this_map)
    if id50[i] == users[2]:
        folium.CircleMarker(location=[la50[i], lo50[i]],
                            radius=2,
                            weight=2, color="blue").add_to(this_map)
    if id50[i] == users[3]:
        folium.CircleMarker(location=[la50[i], lo50[i]],
                            radius=2,
                            weight=2, color="green").add_to(this_map)
    if id50[i] == users[4]:
        folium.CircleMarker(location=[la50[i], lo50[i]],
                            radius=2,
                            weight=2, color="yellow").add_to(this_map)
#Set the zoom to the maximum possible
this_map.fit_bounds(this_map.get_bounds())

#Save the map to an HTML file
this_map.save('individual.html')

