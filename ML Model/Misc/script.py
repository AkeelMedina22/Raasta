import os

import firebase_admin
import numpy as np
import tensorflow as tf
from firebase_admin import credentials, db
from geopy.distance import great_circle
from scipy.interpolate import CubicSpline
from shapely.geometry import MultiPoint
from sklearn.cluster import DBSCAN


def get_data():

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
                csx = CubicSpline(
                    np.arange(60), np.array(without_labels)[:, 0])
                csy = CubicSpline(
                    np.arange(60), np.array(without_labels)[:, 1])
                csz = CubicSpline(
                    np.arange(60), np.array(without_labels)[:, 2])
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
    reconstructed_model = tf.keras.models.load_model("Raasa_Model", compile=False)
    reconstructed_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam())

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


def new_locs():


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

    # this_map = folium.Map(prefer_canvas=True)

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

    kms_per_radian = 6371.0088
    epsilon = 0.075 / kms_per_radian

    pclusters = DBSCAN(eps=epsilon, min_samples=1, metric='haversine',
                       algorithm='ball_tree').fit(np.radians(potholes))
    plabels = pclusters.labels_

    sclusters = DBSCAN(eps=epsilon, min_samples=1, metric='haversine',
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

    pref = db.reference("/pothole-locations/")
    sref = db.reference("/speedbreaker-locations/")
    pref.delete()
    sref.delete()
    pref = db.reference("/pothole-locations/")
    sref = db.reference("/speedbreaker-locations/")

    for i in range(len(centermost_points)):
        sref.push().set(
            {"latitude": centermost_points[i][0], "longitude": centermost_points[i][1]})

    for i in range(len(p_centermost_points)):
        pref.push().set(
            {"latitude": p_centermost_points[i][0], "longitude": p_centermost_points[i][1]})

merge()
