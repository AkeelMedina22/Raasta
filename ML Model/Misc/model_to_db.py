import os

import firebase_admin
import folium
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from firebase_admin import credentials, db
from geopy.distance import great_circle
from numpy import dstack
from pandas import read_csv
from scipy.interpolate import CubicSpline, splev, splrep
from shapely.geometry import MultiPoint
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS

sns.set()


def get_data():

    new_acc_x = []
    new_acc_y = []
    new_acc_z = []

    GOOGLE_APPLICATION_CREDENTIALS = "raasta-c542d-firebase-adminsdk-5v3pa-94bf94e3fb.json"

    cred_obj = credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS)
    default_app = firebase_admin.initialize_app(cred_obj, {
                                                'databaseURL': "https://raasta-c542d-default-rtdb.asia-southeast1.firebasedatabase.app/"})

    ref = db.reference("/unlabelled-data/")
    session_data = list(ref.get().values())

    accelerometer_x = []
    accelerometer_y = []
    accelerometer_z = []
    gyroscope_x = []
    gyroscope_y = []
    gyroscope_z = []
    timestamps = []
    latitude = []
    longitude = []

    for session in session_data:

        for key in sorted(session):

            try:
                timestamps.append(session[key]['timestamps'])
            except KeyError:
                timestamps.append(0)

            try:
                accelerometer_x.append(float(session[key]['accelerometer-x']))
            except KeyError:
                accelerometer_x.append(0.0)

            try:
                accelerometer_y.append(float(session[key]['accelerometer-y']))
            except KeyError:
                accelerometer_y.append(0.0)

            try:
                accelerometer_z.append(float(session[key]['accelerometer-z']))
            except KeyError:
                accelerometer_z.append(0.0)

            try:
                gyroscope_x.append(float(session[key]['gyroscope-x']))
            except KeyError:
                gyroscope_x.append(0.0)

            try:
                gyroscope_y.append(float(session[key]['gyroscope-y']))
            except KeyError:
                gyroscope_y.append(0.0)

            try:
                gyroscope_z.append(float(session[key]['gyroscope-z']))
            except KeyError:
                gyroscope_z.append(0.0)

            try:
                latitude.append(float(session[key]['latitude']))
            except KeyError:
                latitude.append(0)

            try:
                longitude.append(float(session[key]['longitude']))
            except KeyError:
                longitude.append(0)

    # for index in range(len(accelerometer_x)):
    #     # if accelerometer_x[index] == 0.0 or accelerometer_y[index] == 0.0 or accelerometer_z[index] == 0.0:
    #     #     print(accelerometer_x[index], accelerometer_y[index], accelerometer_z[index])

    #     # if accelerometer_x[index] == 0.0:
    #     #     accelerometer_x[index] = 1e-6
    #     # if accelerometer_y[index] == 0.0:
    #     #     accelerometer_y[index] == 1e-6
    #     # if accelerometer_z[index] == 0.0:
    #     #     accelerometer_z[index] == 1e-6

    #     alpha = np.arctan(accelerometer_y[index]/accelerometer_z[index])
    #     beta = np.arctan((-accelerometer_x[index]) / np.sqrt((accelerometer_y[index]**2) + (accelerometer_z[index]**2)))

    #     # OLD RE-ORIENTATION
    #     # new_x = (np.cos(beta) * accelerometer_x[index]) + (np.sin(beta) * np.sin(alpha) * accelerometer_y[index]) + (np.cos(alpha) * np.sin(beta) * accelerometer_z[index])
    #     # new_y = (np.cos(alpha) * accelerometer_y[index]) - (np.sin(alpha) * accelerometer_z[index])
    #     # new_z = (-np.sin(beta) * accelerometer_x[index]) + (np.cos(beta) * np.sin(alpha) * accelerometer_y[index]) + (np.cos(beta) * np.cos(alpha) * accelerometer_z[index])

    #     # new_acc_x.append(new_x)
    #     # new_acc_y.append(new_y)
    #     # new_acc_z.append(new_z)

    #     # NEW ORIENTATION
    #     acc = np.array([accelerometer_x[index], accelerometer_y[index], accelerometer_z[index]])

    #     R_x = np.array([[1, 0, 0],
    #         [0, np.cos(alpha), -np.sin(alpha)],
    #         [0, np.sin(alpha), np.cos(alpha)]])

    #     R_y = np.array([[np.cos(beta), 0, np.sin(beta)],
    #         [0, 1, 0],
    #         [-np.sin(beta), 0, np.cos(beta)]])

    #     result = np.dot(acc, np.dot(R_x, R_y))

    #     gamma = np.arctan(result[0] / result[1])

    #     R_z = np.array([[np.cos(gamma), -np.sin(gamma), 0],
    #         [np.sin(gamma), np.cos(gamma), 0],
    #         [0, 0, 1]])

    #     new_val = np.dot(result, R_z)

    #     new_acc_x.append(new_val[0])
    #     new_acc_y.append(new_val[1])
    #     new_acc_z.append(new_val[2])

    # accelerometer_x = new_acc_x
    # accelerometer_y = new_acc_y
    # accelerometer_z = new_acc_z
    return [[a, b, c, d, e, f] for a, b, c, d, e, f in zip(accelerometer_x, accelerometer_y, accelerometer_z, gyroscope_x, gyroscope_y, gyroscope_z)], longitude, latitude


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
                # csx = CubicSpline(
                #     np.arange(60), np.array(without_labels)[:, 0])
                # csy = CubicSpline(
                #     np.arange(60), np.array(without_labels)[:, 1])
                # csz = CubicSpline(
                #     np.arange(60), np.array(without_labels)[:, 2])
                # newx = csx(np.arange(0, 60, 0.5))
                # newy = csy(np.arange(0, 60, 0.5))
                # newz = csz(np.arange(0, 60, 0.5))
                # new = np.vstack((newx, newy, newz)).T
                window.append(without_labels)
                window_loc.append([latitude[i+30], longitude[i+30]])

        data = np.array(window, dtype=object)
        locs = np.array(window_loc, dtype=object)

        return data, locs

    data, longitude, latitude = get_data()
    X, locs = train_test_split(data, longitude, latitude)

    print(X.shape)
    X = np.asarray(X).astype('float32')

    # It can be used to reconstruct the model identically.
    reconstructed_model = tf.keras.models.load_model("Raasta_Model_new")

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


# GOOGLE_APPLICATION_CREDENTIALS="raasta-c542d-firebase-adminsdk-5v3pa-94bf94e3fb.json"

# cred_obj = credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS)
# default_app = firebase_admin.initialize_app(cred_obj, {'databaseURL':"https://raasta-c542d-default-rtdb.asia-southeast1.firebasedatabase.app/"})

def new_locs():

    # pref = db.reference("/pothole-locations/")
    # bref = db.reference("/badroad-locations/")
    # sref = db.reference("/speedbreaker-locations/")

    data = get_locs()

    potholes = data[1]
    speedbreakers = data[2]

    kms_per_radian = 6371.0088
    epsilon = 0.075 / kms_per_radian

    pclusters = DBSCAN(eps=epsilon, min_samples=2, metric='haversine',
                       algorithm='ball_tree').fit(np.radians(potholes))
    plabels = pclusters.labels_

    sclusters = DBSCAN(eps=epsilon, min_samples=2, metric='haversine',
                       algorithm='ball_tree').fit(np.radians(speedbreakers))
    slabels = sclusters.labels_

    final_s = []
    final_p = []

    this_map = folium.Map(prefer_canvas=True)

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

    # for i in range(len(slabels)):
    #     if slabels[i] == -1:
    #         folium.CircleMarker(location=[speedbreakers[i][0], speedbreakers[i][1]],
    #                             radius=6,
    #                             weight=6, color="grey").add_to(this_map)
    #     else:
    #         folium.CircleMarker(location=[speedbreakers[i][0], speedbreakers[i][1]],
    #                             radius=6,
    #                             weight=6, color="black").add_to(this_map)
    # for i in centermost_points:
    #     folium.CircleMarker(location=i,
    #                         radius=8,
    #                         weight=8, color="green").add_to(this_map)

    # for i in range(len(plabels)):
    #     if plabels[i] == -1:
    #         folium.CircleMarker(location=[potholes[i][0], potholes[i][1]],
    #                             radius=6,
    #                             weight=6, color="grey").add_to(this_map)
    #     else:
    #         folium.CircleMarker(location=[potholes[i][0], potholes[i][1]],
    #                             radius=6,
    #                             weight=6, color="black").add_to(this_map)
    # for i in p_centermost_points:
    #     folium.CircleMarker(location=i,
    #                         radius=8,
    #                         weight=8, color="red").add_to(this_map)

    # for i in range(len(slabels)):
    #     if slabels[i] == -1:
    #         folium.CircleMarker(location=[speedbreakers[i][0], speedbreakers[i][1]],
    #                     radius=6,
    #                     weight=6, color="black").add_to(this_map)
    #     elif slabels[i] not in sunique:
    #         sunique.add(slabels[i])
    #         final_s.append((speedbreakers[i][0], speedbreakers[i][1]))
    #         # pref.push().set({"latitude": speedbreakers[i][0], "longitude": speedbreakers[i][1], "label": "Speedbreaker"})
    #         # sref.push().set({"latitude": speedbreakers[i][0], "longitude": speedbreakers[i][1]})
    #         folium.CircleMarker(location=[speedbreakers[i][0], speedbreakers[i][1]],
    #                     radius=6,
    #                     weight=6, color="green").add_to(this_map)

    # for i in range(len(plabels)):
    #     if plabels[i] == -1:
    #         continue
    #     elif plabels[i] not in punique:
    #         punique.add(plabels[i])
    #         final_p.append((potholes[i][0], potholes[i][1]))
    #         # pref.push().set({"latitude": potholes[i][0], "longitude": potholes[i][1]})
    #         folium.CircleMarker(location=[potholes[i][0], potholes[i][1]],
    #                     radius=6,
    #                     weight=6, color="red").add_to(this_map)

    # Set the zoom to the maximum possible
    this_map.fit_bounds(this_map.get_bounds())

    # Save the map to an HTML file
    this_map.save('folium_visualization_core.html')

    return p_centermost_points, centermost_points


def old_locs():

    pref = db.reference("/pothole-locations/")
    sref = db.reference("/speedbreaker-locations/")

    plocs = []
    slocs = []

    session_data = list(pref.get().values())

    for pothole in session_data:

        plocs.append((pothole['latitude'], pothole['longitude']))

    session_data = list(sref.get().values())

    for speedbreaker in session_data:

        slocs.append((speedbreaker['latitude'], speedbreaker['longitude']))

    return plocs, slocs


def merge():

    new = new_locs()
    old = old_locs()

    potholes = np.concatenate((new[0], old[0]))
    speedbreakers = np.concatenate((new[1], old[1]))
    # potholes = new[0]
    # speedbreakers = new[1]

    kms_per_radian = 6371.0088
    epsilon = 0.125 / kms_per_radian

    pclusters = DBSCAN(eps=epsilon, min_samples=1, metric='haversine',
                       algorithm='ball_tree').fit(np.radians(potholes))
    plabels = pclusters.labels_

    sclusters = DBSCAN(eps=epsilon, min_samples=1, metric='haversine',
                       algorithm='ball_tree').fit(np.radians(speedbreakers))
    slabels = sclusters.labels_

    # this_map = folium.Map(prefer_canvas=True)

    # for _ in new:
    #     for i in _:
    #         folium.CircleMarker(location=[i['latitude'], i['longitude']],
    #                     radius=6,
    #                     weight=6, color="red").add_to(this_map)
    # for _ in old:
    #     for i in _:
    #         folium.CircleMarker(location=[i['latitude'], i['longitude']],
    #                     radius=6,
    #                     weight=6, color="black").add_to(this_map)

    # punique = set()
    # sunique = set()

    # for i in range(len(slabels)):
    #     if slabels[i] == -1:
    #         continue
    #     elif slabels[i] not in sunique:
    #         sunique.add(slabels[i])
    #         # final_s.append({"latitude": speedbreakers[i][0], "longitude": speedbreakers[i][1]})
    #         # pref.push().set({"latitude": speedbreakers[i][0], "longitude": speedbreakers[i][1], "label": "Speedbreaker"})
    #         # sref.push().set({"latitude": speedbreakers[i][0], "longitude": speedbreakers[i][1]})
    #         folium.CircleMarker(location=[speedbreakers[i][0], speedbreakers[i][1]],
    #                             radius=6,
    #                             weight=6, color="green").add_to(this_map)

    # for i in range(len(plabels)):
    #     if plabels[i] == -1:
    #         continue
    #     elif plabels[i] not in punique:
    #         punique.add(plabels[i])
    #         # final_p.append({"latitude": potholes[i][0], "longitude": potholes[i][1]})
    #         # pref.push().set({"latitude": potholes[i][0], "longitude": potholes[i][1]})
    #         folium.CircleMarker(location=[potholes[i][0], potholes[i][1]],
    #                             radius=6,
    #                             weight=6, color="red").add_to(this_map)

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

    pref = db.reference("/pothole-locations/")
    sref = db.reference("/speedbreaker-locations/")
    pref.delete()
    sref.delete()
    pref = db.reference("/pothole-locations/")
    sref = db.reference("/speedbreaker-locations/")
    print("created")

    for i in range(len(centermost_points)):
        # final_s.append((speedbreakers[i][0], speedbreakers[i][1]))
        # pref.push().set({"latitude": speedbreakers[i][0], "longitude": speedbreakers[i][1], "label": "Speedbreaker"})
        sref.push().set(
            {"latitude": centermost_points[i][0], "longitude": centermost_points[i][1]})
        # print(i, "/"+str(len(centermost_points)))
        # folium.CircleMarker(location=[centermost_points[i][0], centermost_points[i][1]],
        #             radius=6,
        #             weight=6, color="green").add_to(this_map)

    for i in range(len(p_centermost_points)):
        pref.push().set(
            {"latitude": p_centermost_points[i][0], "longitude": p_centermost_points[i][1]})
        # print(i, "/"+str(len(p_centermost_points)))
        # folium.CircleMarker(location=[p_centermost_points[i][0], p_centermost_points[i][1]],
        #             radius=6,
        #             weight=6, color="red").add_to(this_map)

    # # Set the zoom to the maximum possible
    # this_map.fit_bounds(this_map.get_bounds())

    # # Save the map to an HTML file
    # this_map.save('folium_visualization_merged_new.html')


merge()
