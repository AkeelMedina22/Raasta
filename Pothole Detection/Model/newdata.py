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

def filter(data, fs=10, fc=2.0, order=11):
    # fc = frequency cutoff
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(order, w, 'highpass', analog=False)
    return signal.filtfilt(b, a, data)


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

    with open('raasta-c542d-default-rtdb-sensor-data-export.json', 'r') as f:
    # with open('raasta-c542d-default-rtdb-sensor-data2-export.json', 'r') as f:

        data = json.load(f)
        for key in data:
            for timestamp in data[key]:
                for l in data[key][timestamp]:
                    if l == "accelerometer-x":
                        accelerometer_x.append(float(data[key][timestamp][l]))

                    elif l == "accelerometer-y":
                        accelerometer_y.append(float(data[key][timestamp][l]))
           
                    elif l == "accelerometer-z":
                        accelerometer_z.append(float(data[key][timestamp][l]))
                
                    elif l == "gyroscope-x":
                        gyroscope_x.append(float(data[key][timestamp][l]))
  
                    elif l == "gyroscope-y":
                        gyroscope_y.append(float(data[key][timestamp][l]))
  
                    elif l == "gyroscope-z":
                        gyroscope_z.append(float(data[key][timestamp][l]))

                    elif l == "latitude":
                        latitude.append(float(data[key][timestamp][l]))

                    elif l == "longitude":
                        longitude.append(float(data[key][timestamp][l]))

                    elif l == "timestamp":
                        timestamps.append(data[key][timestamp][l])

                    elif l == "label":
                        labels.append(data[key][timestamp][l])

    for index in range(len(accelerometer_x)):
        # if accelerometer_x[index] == 0.0 or accelerometer_y[index] == 0.0 or accelerometer_z[index] == 0.0:
        #     print(accelerometer_x[index], accelerometer_y[index], accelerometer_z[index])

        # if accelerometer_x[index] == 0.0:
        #     accelerometer_x[index] = 1e-6
        # if accelerometer_y[index] == 0.0:
        #     accelerometer_y[index] == 1e-6
        # if accelerometer_z[index] == 0.0:
        #     accelerometer_z[index] == 1e-6

        alpha = np.arctan(accelerometer_y[index]/accelerometer_z[index])
        beta = np.arctan((-accelerometer_x[index]) / np.sqrt((accelerometer_y[index]**2) + (accelerometer_z[index]**2)))

        # OLD RE-ORIENTATION
        # new_x = (np.cos(beta) * accelerometer_x[index]) + (np.sin(beta) * np.sin(alpha) * accelerometer_y[index]) + (np.cos(alpha) * np.sin(beta) * accelerometer_z[index])
        # new_y = (np.cos(alpha) * accelerometer_y[index]) - (np.sin(alpha) * accelerometer_z[index])
        # new_z = (-np.sin(beta) * accelerometer_x[index]) + (np.cos(beta) * np.sin(alpha) * accelerometer_y[index]) + (np.cos(beta) * np.cos(alpha) * accelerometer_z[index])

        # new_acc_x.append(new_x)
        # new_acc_y.append(new_y)
        # new_acc_z.append(new_z)

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


    accelerometer_x = filter(new_acc_x)
    accelerometer_y = filter(new_acc_y)
    accelerometer_z = filter(new_acc_z)
    gyroscope_x = filter(gyroscope_x)
    gyroscope_y = filter(gyroscope_y)
    gyroscope_z = filter(gyroscope_z)

    return [[[a,b,c,d,e,f],label] for a,b,c,d,e,f,label in zip(accelerometer_x, accelerometer_y, accelerometer_z, gyroscope_x, gyroscope_y, gyroscope_z, labels)], longitude, latitude
