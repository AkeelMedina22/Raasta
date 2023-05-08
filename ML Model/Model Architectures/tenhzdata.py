import firebase_admin
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import itertools
from firebase_admin import credentials
from firebase_admin import db
from scipy import signal
import math
import json

# def filter(data, fs=10, fc=2.0, order=11):
#     # fc = frequency cutoff
#     w = fc / (fs / 2) # Normalize the frequency
#     b, a = signal.butter(order, w, 'highpass', analog=False)
#     return signal.filtfilt(b, a, data)


def get_data():

    accelerometer_x = []
    accelerometer_y = []
    accelerometer_z = []
    gyroscope_x = []
    gyroscope_y = []
    gyroscope_z = []
    timestamps = []
    latitude = []
    longitude = []
    labels = []

    new_acc_x = []
    new_acc_y = []
    new_acc_z = []

    GOOGLE_APPLICATION_CREDENTIALS="raasta-c542d-firebase-adminsdk-5v3pa-94bf94e3fb.json"

    cred_obj = credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS)
    default_app = firebase_admin.initialize_app(cred_obj, {'databaseURL':"https://raasta-c542d-default-rtdb.asia-southeast1.firebasedatabase.app/"})

    ref = db.reference("/sensor-data/")
    session_data = list(ref.get().values())

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
                accelerometer_z.append(10.0)
            
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
            
            try:
                labels.append(session[key]['label'])
            except KeyError:
                labels.append("")

    for index in range(len(accelerometer_x)):

        alpha = np.arctan(accelerometer_y[index]/accelerometer_z[index])
        beta = np.arctan((-accelerometer_x[index]) / np.sqrt((accelerometer_y[index]**2) + (accelerometer_z[index]**2)))

        # NEW ORIENTATION
        acc = np.array([accelerometer_x[index], accelerometer_y[index], accelerometer_z[index]])

        R_x = np.array([[1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)]])
        
        R_y = np.array([[np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)]])
        
        result = np.dot(acc, np.dot(R_x, R_y))
        
        gamma = np.arctan(result[0] / result[1])
        
        R_z = np.array([[np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1]])
        
        new_val = np.dot(result, R_z)

        new_acc_x.append(new_val[0])
        new_acc_y.append(new_val[1])
        new_acc_z.append(new_val[2])

    
    accelerometer_x = new_acc_x
    accelerometer_y = new_acc_y
    accelerometer_z = new_acc_z

    return [[[a,b,c,d,e,f],label] for a,b,c,d,e,f,label in zip(accelerometer_x, accelerometer_y, accelerometer_z, gyroscope_x, gyroscope_y, gyroscope_z, labels)], longitude, latitude

if __name__ == "__main__":
    get_data()
