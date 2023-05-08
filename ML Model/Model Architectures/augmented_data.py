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
import synthia as syn

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

    with open('raasta-c542d-default-rtdb-sensor-data-export.json', 'r') as f:

        data = json.load(f)
        for key in data:
            tax, tay, taz, tgx, tgy, tgz = [], [], [], [], [], []
            for timestamp in data[key]:
                for l in data[key][timestamp]:
                    if l == "accelerometer-x":
                        tax.append(float(data[key][timestamp][l]))

                    elif l == "accelerometer-y":
                        tay.append(float(data[key][timestamp][l]))
           
                    elif l == "accelerometer-z":
                        taz.append(float(data[key][timestamp][l]))
                
                    elif l == "gyroscope-x":
                        tgx.append(float(data[key][timestamp][l]))
  
                    elif l == "gyroscope-y":
                        tgy.append(float(data[key][timestamp][l]))
  
                    elif l == "gyroscope-z":
                        tgz.append(float(data[key][timestamp][l]))

                    elif l == "latitude":
                        latitude.append(float(data[key][timestamp][l]))

                    elif l == "longitude":
                        longitude.append(float(data[key][timestamp][l]))

                    elif l == "timestamp":
                        timestamps.append(data[key][timestamp][l])

                    elif l == "label":
                        labels.append(data[key][timestamp][l])

            dataa = [tax, tay, taz, tgx, tgy, tgz]
            for i in range(6):
                generator = syn.CopulaDataGenerator()
                generator.fit(np.array(dataa[i]).reshape(1, -1), copula=syn.IndependenceCopula())
                samples = generator.generate(n_samples=10,  uniformization_ratio=0.5)   
                for j in samples:
                    for k in j:
                        if i == 0:
                            accelerometer_x.append(k)
                        elif i == 1:
                            accelerometer_y.append(k)
                        elif i == 2:
                            accelerometer_z.append(k)
                        elif i == 3:
                            gyroscope_x.append(k)
                        elif i == 4:
                            gyroscope_y.append(k)
                        elif i == 5:
                            gyroscope_z.append(k)
            

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

    # accelerometer50_x = []
    # accelerometer50_y = []
    # accelerometer50_z = []
    # gyroscope50_x = []
    # gyroscope50_y = []
    # gyroscope50_z = []
    # timestamps50 = []
    # latitude50 = []
    # longitude50 = []
    # labels50 = []

    # new_acc_x50 = []
    # new_acc_y50 = []
    # new_acc_z50 = []

    # with open('raasta-c542d-default-rtdb-sensor-data2-export.json', 'r') as f:

    #     data = json.load(f)
    #     for key in data:
    #         tax, tay, taz, tgx, tgy, tgz = [], [], [], [], [], []
    #         for timestamp in data[key]:
    #             for l in data[key][timestamp]:
    #                 if l == "accelerometer-x":
    #                     tax.append(float(data[key][timestamp][l]))

    #                 elif l == "accelerometer-y":
    #                     tay.append(float(data[key][timestamp][l]))
           
    #                 elif l == "accelerometer-z":
    #                     taz.append(float(data[key][timestamp][l]))
                
    #                 elif l == "gyroscope-x":
    #                     tgx.append(float(data[key][timestamp][l]))
  
    #                 elif l == "gyroscope-y":
    #                     tgy.append(float(data[key][timestamp][l]))
  
    #                 elif l == "gyroscope-z":
    #                     tgz.append(float(data[key][timestamp][l]))

    #                 elif l == "latitude":
    #                     latitude50.append(float(data[key][timestamp][l]))

    #                 elif l == "longitude":
    #                     longitude50.append(float(data[key][timestamp][l]))

    #                 elif l == "timestamp":
    #                     timestamps50.append(data[key][timestamp][l])

    #                 elif l == "label":
    #                     labels50.append(data[key][timestamp][l])

    #         dataa = [tax, tay, taz, tgx, tgy, tgz]
    #         for i in range(6):
    #             generator = syn.CopulaDataGenerator()
    #             generator.fit(np.array(dataa[i]).reshape(1, -1), copula=syn.IndependenceCopula())
    #             samples = generator.generate(n_samples=10,  uniformization_ratio=0.5)   
    #             for j in samples:
    #                 for k in j:
    #                     if i == 0:
    #                         accelerometer50_x.append(k)
    #                     elif i == 1:
    #                         accelerometer50_y.append(k)
    #                     elif i == 2:
    #                         accelerometer50_z.append(k)
    #                     elif i == 3:
    #                         gyroscope50_x.append(k)
    #                     elif i == 4:
    #                         gyroscope50_y.append(k)
    #                     elif i == 5:
    #                         gyroscope50_z.append(k)

    # for index in range(len(accelerometer50_x)):

    #     alpha = np.arctan(accelerometer50_y[index]/accelerometer50_z[index])
    #     beta = np.arctan((-accelerometer50_x[index]) / np.sqrt((accelerometer50_y[index]**2) + (accelerometer50_z[index]**2)))

    #     # NEW ORIENTATION
    #     acc = np.array([accelerometer50_x[index], accelerometer50_y[index], accelerometer50_z[index]])

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

    #     new_acc_x50.append(new_val[0])
    #     new_acc_y50.append(new_val[1])
    #     new_acc_z50.append(new_val[2])
    
    # _x = signal.resample(new_acc_x50, len(new_acc_x50)//5)
    # _y = signal.resample(new_acc_y50, len(new_acc_x50)//5)
    # _z = signal.resample(new_acc_z50, len(new_acc_x50)//5)
    # _gyx = signal.resample(gyroscope50_x, len(new_acc_x50)//5)
    # _gyy = signal.resample(gyroscope50_y, len(new_acc_x50)//5)
    # _gyz = signal.resample(gyroscope50_z, len(new_acc_x50)//5)

    # new_labels = []
    # new_lat = []
    # new_long = []
    # for i in range(0, len(accelerometer50_x), 5):
    #     new_labels.append(labels50[i])
    #     new_lat.append(latitude50[i])
    #     new_long.append(longitude50[i])

    # accelerometer_x = np.hstack((new_acc_x, _x))
    # accelerometer_y = np.hstack((new_acc_y, _y))
    # accelerometer_z = np.hstack((new_acc_z, _z)) 
    # gyroscope_x = np.hstack((gyroscope_x, _gyx))
    # gyroscope_y = np.hstack((gyroscope_y, _gyy))
    # gyroscope_z = np.hstack((gyroscope_z, _gyz))
    # labels = np.hstack((labels, new_labels))
    # longitude = np.hstack((longitude, new_long))
    # latitude = np.hstack((latitude, new_lat))   

    return [[[a,b,c,d,e,f],label] for a,b,c,d,e,f,label in zip(accelerometer_x, accelerometer_y, accelerometer_z, gyroscope_x, gyroscope_y, gyroscope_z, labels)], longitude, latitude

if __name__ == "__main__":
    get_data()
