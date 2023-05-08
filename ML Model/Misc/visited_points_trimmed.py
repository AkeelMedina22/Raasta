import os

import firebase_admin
import folium
import numpy as np
from firebase_admin import credentials, db


def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


GOOGLE_APPLICATION_CREDENTIALS = "raasta-c542d-firebase-adminsdk-5v3pa-94bf94e3fb.json"

cred_obj = credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS)
default_app = firebase_admin.initialize_app(cred_obj, {
                                            'databaseURL': "https://raasta-c542d-default-rtdb.asia-southeast1.firebasedatabase.app/"})

ref = db.reference("/unlabelled-data/")
session_data = list(ref.get().values())

latitude = []
longitude = []

# this_map = folium.Map(prefer_canvas=True)

for session in session_data:
        
    temp1 = []
    temp2 = []

    old_lat = 0.0
    old_long = 0.0

    for key in sorted(session):

        if old_lat != float(session[key]['latitude']) and old_long != float(session[key]['longitude']): 
            temp1.append(float(session[key]['latitude']))
            temp2.append(float(session[key]['longitude']))
            old_lat = float(session[key]['latitude'])
            old_long = float(session[key]['longitude'])

    latitude.append(temp1)
    longitude.append(temp2)

rawcount = 0

for i in latitude:
    for j in i:
        rawcount += 1

all_points = []

for i,j in zip(latitude, longitude):

    try:
        point_a = np.array([i[0], j[0]])
        point_b = np.array([i[1], j[1]])

        points = [point_a, point_b]

        for k in range(2, len(i)):

            angle = 0.0

            point_c = np.array([i[k], j[k]])

            angle = angle_between(point_b-point_a, point_c-point_b)

            if np.abs(angle) > 1.5:

                points.append(point_c)

            point_a = np.array([i[k-1], j[k-1]])
            point_b = np.array([i[k], j[k]])

        if len(points) > 2:
            all_points.append(points)

    except:
        pass
        

dbref = db.reference("/visited/")
final = {}
for j in range(len(all_points)):
    dic = {}
    for i in range(0, len(all_points[j])):
        dic.update({str(i): {"latitude": all_points[j][i][0], "longitude": all_points[j][i][1]}})
        #folium.PolyLine([[all_points[j][i-1][0], all_points[j][i-1][1]], [all_points[j][i][0], all_points[j][i][1]]]).add_to(this_map)
    print(j, len(all_points[j]), len(all_points))
    final.update({j: dic})

dbref.push().set(final)

# # Set the zoom to the maximum possible
# this_map.fit_bounds(this_map.get_bounds())

# # Save the map to an HTML file
# this_map.save('clustered_points_full.html')