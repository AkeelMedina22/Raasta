import firebase_admin
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import itertools
from firebase_admin import credentials
from firebase_admin import db
from scipy import signal

def filter(data, fs=10, fc=2.5, order=11):
    # fc = frequency cutoff
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(order, w, 'highpass', analog=False)
    return signal.filtfilt(b, a, data)

GOOGLE_APPLICATION_CREDENTIALS="raasta-c542d-firebase-adminsdk-5v3pa-94bf94e3fb.json"

cred_obj = credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS)
default_app = firebase_admin.initialize_app(cred_obj, {'databaseURL':"https://raasta-c542d-default-rtdb.asia-southeast1.firebasedatabase.app/"})

ref = db.reference("/sensor-data2/0a05c04d-c371-4b20-8527-337202bd82f8/")
session_data = ref.get()

accelerometer_data = []
gyroscope_data = []
accelerometer_x = []
accelerometer_y = []
accelerometer_z = []
gyroscope_x = []
gyroscope_y = []
gyroscope_z = []
timestamps = []
latitude = []
longitude = []

colors = []
include = ['Pothole', 'Bad Road', 'Speedbreaker']
color_map = {0: 'darkslategray', 1: 'goldenrod'}

for key in sorted(session_data):

    try:
        if session_data[key]['label'] in include:
            colors.append(0)
        else:
            colors.append(1)

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
            gyroscope_x.append(float(session_data[key]['gyroscope-x']))
        except KeyError:
            gyroscope_x.append(0.0)

        try:
            gyroscope_y.append(float(session_data[key]['gyroscope-y']))
        except KeyError:
            gyroscope_y.append(0.0)

        try:
            gyroscope_z.append(float(session_data[key]['gyroscope-z']))
        except KeyError:
            gyroscope_z.append(0.0)

        try:
            latitude.append(float(session_data[key]['latitude']))
        except KeyError:
            latitude.append(0)

        try:
            longitude.append(float(session_data[key]['longitude']))
        except KeyError:
            longitude.append(0)
    except:
        pass

filt_acx = accelerometer_x
filt_acy = accelerometer_y
filt_acz = accelerometer_z
filt_gyx = gyroscope_x
filt_gyy = gyroscope_y
filt_gyz = gyroscope_z

# filt_acx = filter(accelerometer_x)
# filt_acy = filter(accelerometer_y)
# filt_acz = filter(accelerometer_z)
# filt_gyx = filter(gyroscope_x)
# filt_gyy = filter(gyroscope_y)
# filt_gyz = filter(gyroscope_z)

accelerometer_data = [(i,j,k) for i,j,k in zip(filt_acx, filt_acy, filt_acz)]
gyroscope_data = [(i,j,k) for i,j,k in zip(filt_gyx, filt_gyy, filt_gyz)]

# latitude = list(filter(lambda num: num != 0, latitude))
# longitude = list(filter(lambda num: num != 0, longitude))

c = np.arange(len(latitude))

fig = plt.figure()
ax01 = fig.add_subplot(3,3,2)
ax02 = fig.add_subplot(3,3,4)
ax03 = fig.add_subplot(3,3,5)
ax04 = fig.add_subplot(3,3,6)
ax05 = fig.add_subplot(3,3,7)
ax06 = fig.add_subplot(3,3,8)
ax07 = fig.add_subplot(3,3,9)

ax01.set_title("GPS")
ax01.scatter(longitude, latitude, c=c, cmap='viridis', s=1)

ax02.set_facecolor((0.0, 0.5, 1.0, 0.2))
ax03.set_facecolor((0.0, 0.5, 1.0, 0.2))
ax04.set_facecolor((0.0, 0.5, 1.0, 0.2))
ax05.set_facecolor((0.0, 0.5, 1.0, 0.2))
ax06.set_facecolor((0.0, 0.5, 1.0, 0.2))
ax07.set_facecolor((0.0, 0.5, 1.0, 0.2))

for i in range(len(gyroscope_data)):
    if colors[i] == 0:
        ax02.axvspan(i-0.5, i+0.5, facecolor='salmon', alpha=1.0)
        ax03.axvspan(i-0.5, i+0.5, facecolor='salmon', alpha=1.0)
        ax04.axvspan(i-0.5, i+0.5, facecolor='salmon', alpha=1.0)

        ax05.axvspan(i-0.5, i+0.5, facecolor='salmon', alpha=1.0)
        ax06.axvspan(i-0.5, i+0.5, facecolor='salmon', alpha=1.0)
        ax07.axvspan(i-0.5, i+0.5, facecolor='salmon', alpha=1.0)


ax02.set_title("Gyroscope-X")
ax02.plot(range(len(gyroscope_data)), [i[0] for i in gyroscope_data], color='darkslategray')
ax03.set_title("Gyroscope-Y")
ax03.plot(range(len(gyroscope_data)), [i[1] for i in gyroscope_data], color='darkslategray')
ax04.set_title("Gyroscope-Z")
ax04.plot(range(len(gyroscope_data)), [i[2] for i in gyroscope_data], color='darkslategray')

ax05.set_title("Accelerometer-X")
ax05.plot(range(len(accelerometer_data)), [i[0] for i in accelerometer_data], color='darkslateblue')
ax06.set_title("Accelerometer-Y")
ax06.plot(range(len(accelerometer_data)), [i[1] for i in accelerometer_data], color='darkslateblue')
ax07.set_title("Accelerometer-Z")
ax07.plot(range(len(accelerometer_data)), [i[2] for i in accelerometer_data], color='darkslateblue')

plt.tight_layout()
plt.savefig('Visualized_datapoint_filter_2andhalf')
plt.show()
