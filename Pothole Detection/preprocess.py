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

ref = db.reference("/sensor-data/5650eb3a-9dc4-4925-917a-be72f49e3f37/")
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

fs = 10  # Sampling frequency
# Generate the time vector properly
t = np.arange(len(accelerometer_x)) / fs
plt.plot(t, accelerometer_z, label="initial")

fc = 2.5  # Cut-off frequency of the filter
w = fc / (fs / 2) # Normalize the frequency
b, a = signal.butter(11, w, 'lowpass', analog=False)
output = signal.filtfilt(b, a, accelerometer_z)
plt.plot(t, output, label='filtered')

plt.legend()
plt.savefig("Filtered")
plt.show()

spl = splrep(t, accelerometer_z)
t1 = np.arange((len(accelerometer_x))*2) / (fs*2)
acz1 = splev(t1, spl)

print(accelerometer_z[0:10])
print(acz1[0:10])

plt.plot(t1, acz1)
plt.savefig("Interpolated")
plt.show()




