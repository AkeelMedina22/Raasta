import firebase_admin
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import itertools
from firebase_admin import credentials
from firebase_admin import db
from scipy import signal
import folium
def filter(data, fs=10, fc=2.5, order=11):
    # fc = frequency cutoff
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(order, w, 'highpass', analog=False)
    return signal.filtfilt(b, a, data)


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

this_map = folium.Map(prefer_canvas=True)


GOOGLE_APPLICATION_CREDENTIALS="raasta-c542d-firebase-adminsdk-5v3pa-94bf94e3fb.json"
cred_obj = credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS)
default_app = firebase_admin.initialize_app(cred_obj, {'databaseURL':"https://raasta-c542d-default-rtdb.asia-southeast1.firebasedatabase.app/"})

ref = db.reference("/unlabelled-data/")
session_data = list(ref.get().values())


for session in session_data:
    for key in session:
        try:

            latitude.append(float(session[key]['latitude']))
            longitude.append(float(session[key]['longitude']))

            folium.CircleMarker(location=[latitude[-1], longitude[-1]],
                            radius=2,
                            weight=2, color="black").add_to(this_map)
        except:
            pass

ref = db.reference("/sensor-data/")
session_data = list(ref.get().values())


for session in session_data:

    for key in session:

        try:

            latitude.append(float(session[key]['latitude']))
            longitude.append(float(session[key]['longitude']))

            folium.CircleMarker(location=[latitude[-1], longitude[-1]],
                            radius=2,
                            weight=2, color="black").add_to(this_map)
        except:
            pass

ref = db.reference("/sensor-data2/")
session_data = list(ref.get().values())


for session in session_data:
    for key in session:

        try:

            latitude.append(float(session[key]['latitude']))
            longitude.append(float(session[key]['longitude']))

            folium.CircleMarker(location=[latitude[-1], longitude[-1]],
                            radius=2,
                            weight=2, color="black").add_to(this_map)
        except:
            pass

#Set the zoom to the maximum possible
this_map.fit_bounds(this_map.get_bounds())

#Save the map to an HTML file
this_map.save('all_data.html')

