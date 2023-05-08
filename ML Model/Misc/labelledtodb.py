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
    l50 = []
    la50 = []
    lo50 = []


    with open('raasta-c542d-default-rtdb-sensor-data-export.json', 'r') as f:

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
    
    with open('raasta-c542d-default-rtdb-sensor-data2-export.json', 'r') as f:

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


    return labels, longitude, latitude, l50, lo50, la50


def get_locs():

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


    label, longitude, latitude, l50, lo50, la50 = get_data()
    
    pothole_locations = set()
    speedbreaker_locations = set()
    normalroads_locations = set()
    for i in range(0, len(label)-60, 60):
        n = 0
        p = 0
        s = 0
        for j in label[i:i+60]:
            if j == 'Normal Road':
                n += 1
            elif j == 'Pothole' or j == 'Bad Road':
                p += 1
            elif j == 'Speedbreaker':
                s += 1
        if p > 0:
            pothole_locations.add(tuple((latitude[i+30], longitude[i+30])))
        elif s > 0:
            speedbreaker_locations.add(tuple((latitude[i+30], longitude[i+30])))
        elif n > 0:
            normalroads_locations.add(tuple((latitude[i+30], longitude[i+30])))

    for i in range(0, len(l50)-60, 60):
        n = 0
        p = 0
        s = 0
        for j in l50[i:i+60]:
            if j == 'Normal Road':
                n += 1
            elif j == 'Pothole' or j == 'Bad Road':
                p += 1
            elif j == 'Speedbreaker':
                s += 1
        if p > 0:
            pothole_locations.add(tuple((la50[i+30], lo50[i+30])))
        elif s > 0:
            speedbreaker_locations.add(tuple((la50[i+30], lo50[i+30])))
        elif n > 0:
            normalroads_locations.add(tuple((la50[i+30], lo50[i+30])))
        

    normalroads_locations = list(normalroads_locations)
    pothole_locations = list(pothole_locations)
    speedbreaker_locations = list(speedbreaker_locations)

    print(len(normalroads_locations), len(pothole_locations), len(speedbreaker_locations))

    return normalroads_locations, pothole_locations, speedbreaker_locations

GOOGLE_APPLICATION_CREDENTIALS="raasta-c542d-firebase-adminsdk-5v3pa-94bf94e3fb.json"

cred_obj = credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS)
default_app = firebase_admin.initialize_app(cred_obj, {'databaseURL':"https://raasta-c542d-default-rtdb.asia-southeast1.firebasedatabase.app/"})

def new_locs():

    pref = db.reference("/pothole-locations/")
    sref = db.reference("/speedbreaker-locations/")
    pref.delete()
    sref.delete()
    pref = db.reference("/pothole-locations/")
    sref = db.reference("/speedbreaker-locations/")

    data = get_locs()

    potholes = data[1]
    speedbreakers = data[2]

    kms_per_radian = 6371.0088
    epsilon = 0.05 / kms_per_radian

    pclusters = DBSCAN(eps=epsilon, min_samples=2, metric='haversine',
                       algorithm='ball_tree').fit(np.radians(potholes))
    plabels = pclusters.labels_

    sclusters = DBSCAN(eps=epsilon, min_samples=2, metric='haversine',
                       algorithm='ball_tree').fit(np.radians(speedbreakers))
    slabels = sclusters.labels_

    def get_centermost_point(cluster):
        centroid = (MultiPoint(cluster).centroid.x,
                    MultiPoint(cluster).centroid.y)
        centermost_point = min(
            cluster, key=lambda point: great_circle(point, centroid).m)
        return tuple(centermost_point)

    num_clusters = len(set(slabels))-1
    clusters = []
    for i in range(num_clusters):
        temp = []
        for j in range(len(slabels)):
            if slabels[j] == i:
                temp.append(speedbreakers[j])
        clusters.append(temp)

    centermost_points = []
    for i in clusters:
        centermost_points.append(get_centermost_point(i))
    ##########################################################
    num_clusters = len(set(plabels))-1
    clusters = []
    for i in range(num_clusters):
        temp = []
        for j in range(len(plabels)):
            if plabels[j] == i:
                temp.append(potholes[j])
        clusters.append(temp)

    p_centermost_points = []
    for i in clusters:
        p_centermost_points.append(get_centermost_point(i))


    punique = set()
    sunique = set()

    final_s = []
    final_b = []
    final_p = []

    # this_map = folium.Map(prefer_canvas=True)

    # for i in range(len(slabels)):
    #     if slabels[i] == -1:
    #         pass
    #     elif slabels[i] not in sunique:
    #         sunique.add(slabels[i])
    #         # final_s.append((speedbreakers[i][0], speedbreakers[i][1]))
    #         # pref.push().set({"latitude": speedbreakers[i][0], "longitude": speedbreakers[i][1], "label": "Speedbreaker"})
    #         # sref.push().set({"latitude": speedbreakers[i][0], "longitude": speedbreakers[i][1]})
    #         # print(i, "/"+str(len(slabels)))
    #         folium.CircleMarker(location=[speedbreakers[i][0], speedbreakers[i][1]],
    #                     radius=6,
    #                     weight=6, color="green").add_to(this_map)
    #     else:
    #         folium.CircleMarker(location=[potholes[i][0], potholes[i][1]],
    #                     radius=6,
    #                     weight=6, color="black").add_to(this_map)
        

    # for i in range(len(plabels)):
    #     if plabels[i] == -1:
    #         pass
    #     elif plabels[i] not in punique:
    #         punique.add(plabels[i])
    #         # final_p.append((potholes[i][0], potholes[i][1]))
    #         # pref.push().set({"latitude": potholes[i][0], "longitude": potholes[i][1]})
    #         # print(i, "/"+str(len(plabels)))
    #         folium.CircleMarker(location=[potholes[i][0], potholes[i][1]],
    #                     radius=6,
    #                     weight=6, color="red").add_to(this_map)
    #     else:
    #         folium.CircleMarker(location=[potholes[i][0], potholes[i][1]],
    #                     radius=6,
    #                     weight=6, color="black").add_to(this_map)

    for i in range(len(centermost_points)):
        # final_s.append((speedbreakers[i][0], speedbreakers[i][1]))
        # pref.push().set({"latitude": speedbreakers[i][0], "longitude": speedbreakers[i][1], "label": "Speedbreaker"})
        sref.push().set({"latitude": centermost_points[i][0], "longitude": centermost_points[i][1]})
        print(i, "/"+str(len(centermost_points)))
        # folium.CircleMarker(location=[centermost_points[i][0], centermost_points[i][1]],
        #             radius=6,
        #             weight=6, color="green").add_to(this_map)

        

    for i in range(len(p_centermost_points)):
        pref.push().set({"latitude": p_centermost_points[i][0], "longitude": p_centermost_points[i][1]})
        print(i, "/"+str(len(p_centermost_points)))
        # folium.CircleMarker(location=[p_centermost_points[i][0], p_centermost_points[i][1]],
        #             radius=6,
        #             weight=6, color="red").add_to(this_map)

        
    # # #Set the zoom to the maximum possible
    # this_map.fit_bounds(this_map.get_bounds())

    # # #Save the map to an HTML file
    # this_map.save('folium_visualization_all_labelled1.html')
    
    return final_p, final_b, final_s

new_locs()