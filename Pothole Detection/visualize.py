import firebase_admin
import matplotlib.pyplot as plt
import numpy as np
from firebase_admin import credentials
from firebase_admin import db

GOOGLE_APPLICATION_CREDENTIALS="raasta-c542d-firebase-adminsdk-5v3pa-94bf94e3fb.json"

cred_obj = credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS)
default_app = firebase_admin.initialize_app(cred_obj, {'databaseURL':"https://raasta-c542d-default-rtdb.asia-southeast1.firebasedatabase.app/"})

ref = db.reference("/sensor-data/6b59d575-fe4b-4bdc-aa47-1027beb997aa/")
session_data = ref.get()

accelerometer_data = []
gyroscope_data = []
timestamps = []
latitude = []
longitude = []

for key in sorted(session_data):
    
    try:
        timestamps.append(session_data[key]['timestamps'])
    except KeyError:
        timestamps.append(0)

    try:
        accelerometer_data.append((float(session_data[key]['accelerometer-x']), float(session_data[key]['accelerometer-y']), float(session_data[key]['accelerometer-z'])))
    except KeyError:
        accelerometer_data.append((0,0,0))
    
    try:
        gyroscope_data.append((float(session_data[key]['gyroscope-x']), float(session_data[key]['gyroscope-y']), float(session_data[key]['gyroscope-z'])))
    except KeyError:
        gyroscope_data.append((0,0,0))
    
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
plt.show()