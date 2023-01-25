import firebase_admin
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import itertools
from firebase_admin import credentials
from firebase_admin import db
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from math import exp
from scipy import signal
from scipy.interpolate import splev, splrep

GOOGLE_APPLICATION_CREDENTIALS="raasta-c542d-firebase-adminsdk-5v3pa-94bf94e3fb.json"

cred_obj = credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS)
default_app = firebase_admin.initialize_app(cred_obj, {'databaseURL':"https://raasta-c542d-default-rtdb.asia-southeast1.firebasedatabase.app/"})

ref = db.reference("/sensor-data/0f652a22-2694-41ff-98c4-d9eafaafda03/")
session_data = ref.get()

accelerometer_x = []
accelerometer_y = []
accelerometer_z = []
gyroscope_data = []
timestamps = []
latitude = []
longitude = []

colors = []
color_map = {0: 'darkslategray', 1: 'goldenrod'}

for key in sorted(session_data):

    try:
        timestamps.append(session_data[key]['timestamps'])
    except KeyError:
        timestamps.append(0)

    try:
        accelerometer_x.append(float(session_data[key]['accelerometer-x']))
    except KeyError:
        accelerometer_x.append(0.0)

    try:
        accelerometer_y.append(float(session_data[key]['accelerometer-y']))
    except KeyError:
        accelerometer_y.append(0.0)

    try:
        accelerometer_z.append(float(session_data[key]['accelerometer-z']))
    except KeyError:
        accelerometer_z.append(0.0)
    
    try:
        latitude.append(float(session_data[key]['latitude']))
    except KeyError:
        latitude.append(0)

    try:
        longitude.append(float(session_data[key]['longitude']))
    except KeyError:
        longitude.append(0)


latitude = list(filter(lambda num: num != 0, latitude))
longitude = list(filter(lambda num: num != 0, longitude))


def filter(data, fs=10, fc=2, order=11):
    # fc = frequency cutoff
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(order, w, 'lowpass', analog=False)
    return signal.filtfilt(b, a, data)

def resample(data, old_fs, new_fs=2):
    t = np.arange(len(data)) / old_fs
    spl = splrep(t, data)
    t1 = np.arange((len(data))*new_fs) / (fs*new_fs)
    return splev(t1, spl)


fs = 10  # Sampling frequency
new_accz = filter(accelerometer_z)
t = np.arange(len(accelerometer_x)) / fs

plt.plot(t, accelerometer_z, label="initial")
plt.plot(t, new_accz, label='filtered')

plt.legend()
plt.savefig("Filtered")
plt.show()

# generate new sample points on new frequency to plot resampled data



