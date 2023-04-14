from unlabelled_reconstruct import get_locs
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import folium
import firebase_admin
import numpy as np
from firebase_admin import credentials
from firebase_admin import db
from shapely.geometry import MultiPoint
from sklearn import metrics
from geopy.distance import great_circle
import pandas as pd
# GOOGLE_APPLICATION_CREDENTIALS="raasta-c542d-firebase-adminsdk-5v3pa-94bf94e3fb.json"

# cred_obj = credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS)
# default_app = firebase_admin.initialize_app(cred_obj, {'databaseURL':"https://raasta-c542d-default-rtdb.asia-southeast1.firebasedatabase.app/"})

def new_locs():

    # pref = db.reference("/pothole-locations/")
    # bref = db.reference("/badroad-locations/")
    # sref = db.reference("/speedbreaker-locations/")

    data = get_locs()

    potholes = data[1]
    speedbreakers = data[2]

    kms_per_radian = 6371.0088
    epsilon = 0.05 / kms_per_radian


    pclusters = DBSCAN(eps=epsilon, min_samples=1, metric='haversine', algorithm='ball_tree').fit(np.radians(potholes))
    plabels = pclusters.labels_

    sclusters = DBSCAN(eps=epsilon, min_samples=1, metric='haversine', algorithm='ball_tree').fit(np.radians(speedbreakers))
    slabels = sclusters.labels_

    final_s = []
    final_p = []

    this_map = folium.Map(prefer_canvas=True)

    def get_centermost_point(cluster):
        centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
        centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
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
    
    for i in range(len(slabels)):
        if slabels[i] == -1:
            folium.CircleMarker(location=[speedbreakers[i][0], speedbreakers[i][1]],
                        radius=6,
                        weight=6, color="grey").add_to(this_map)
        else:
            folium.CircleMarker(location=[speedbreakers[i][0], speedbreakers[i][1]],
                        radius=6,
                        weight=6, color="black").add_to(this_map)
    for i in centermost_points:
        folium.CircleMarker(location=i,
                        radius=8,
                        weight=8, color="green").add_to(this_map)
    
    for i in range(len(plabels)):
        if plabels[i] == -1:
            folium.CircleMarker(location=[potholes[i][0], potholes[i][1]],
                        radius=6,
                        weight=6, color="grey").add_to(this_map)
        else:
            folium.CircleMarker(location=[potholes[i][0], potholes[i][1]],
                        radius=6,
                        weight=6, color="black").add_to(this_map)
    for i in p_centermost_points:
        folium.CircleMarker(location=i,
                        radius=8,
                        weight=8, color="red").add_to(this_map)

    # for i in range(len(slabels)):
    #     if slabels[i] == -1:
    #         folium.CircleMarker(location=[speedbreakers[i][0], speedbreakers[i][1]],
    #                     radius=6,
    #                     weight=6, color="black").add_to(this_map)
    #     elif slabels[i] not in sunique:
    #         sunique.add(slabels[i])
    #         final_s.append((speedbreakers[i][0], speedbreakers[i][1]))
    #         # pref.push().set({"latitude": speedbreakers[i][0], "longitude": speedbreakers[i][1], "label": "Speedbreaker"})
    #         # sref.push().set({"latitude": speedbreakers[i][0], "longitude": speedbreakers[i][1]})
    #         folium.CircleMarker(location=[speedbreakers[i][0], speedbreakers[i][1]],
    #                     radius=6,
    #                     weight=6, color="green").add_to(this_map)

    # for i in range(len(plabels)):
    #     if plabels[i] == -1:
    #         continue
    #     elif plabels[i] not in punique:
    #         punique.add(plabels[i])
    #         final_p.append((potholes[i][0], potholes[i][1]))
    #         # pref.push().set({"latitude": potholes[i][0], "longitude": potholes[i][1]})
    #         folium.CircleMarker(location=[potholes[i][0], potholes[i][1]],
    #                     radius=6,
    #                     weight=6, color="red").add_to(this_map)
        
    # Set the zoom to the maximum possible
    this_map.fit_bounds(this_map.get_bounds())

    # Save the map to an HTML file
    this_map.save('folium_visualization_core.html')
    
    return final_p, final_s

def old_locs():

    pref = db.reference("/pothole-locations/")
    bref = db.reference("/badroad-locations/")
    sref = db.reference("/speedbreaker-locations/")

    plocs = []
    blocs = []
    slocs = []

    session_data = list(pref.get().values())

    for pothole in session_data:

        plocs.append((pothole['latitude'], pothole['longitude']))
    
    session_data = list(bref.get().values())

    for badroad in session_data:

        blocs.append((badroad['latitude'], badroad['longitude']))

    session_data = list(sref.get().values())

    for speedbreaker in session_data:

        slocs.append((speedbreaker['latitude'], speedbreaker['longitude']))
    
    return plocs, blocs, slocs


def merge():

    new = new_locs()
    old = old_locs()

    potholes = new[0]+old[0]
    badroads = new[1]+old[1]
    speedbreakers = new[2]+old[2]

    pclusters = DBSCAN(eps=0.0005, min_samples=1, p=2).fit(potholes)
    plabels = pclusters.labels_

    bclusters = DBSCAN(eps=0.0005, min_samples=1, p=2).fit(badroads)
    blabels = bclusters.labels_

    sclusters = DBSCAN(eps=0.0005, min_samples=1, p=2).fit(speedbreakers)
    slabels = sclusters.labels_

    this_map = folium.Map(prefer_canvas=True)

    # for _ in new:
    #     for i in _:
    #         folium.CircleMarker(location=[i['latitude'], i['longitude']],
    #                     radius=6,
    #                     weight=6, color="red").add_to(this_map)
    # for _ in old:
    #     for i in _:
    #         folium.CircleMarker(location=[i['latitude'], i['longitude']],
    #                     radius=6,
    #                     weight=6, color="black").add_to(this_map)

    punique = set()
    bunique = set()
    sunique = set()

    for i in range(len(slabels)):
        if slabels[i] == -1:
            continue
        elif slabels[i] not in sunique:
            sunique.add(slabels[i])
            # final_s.append({"latitude": speedbreakers[i][0], "longitude": speedbreakers[i][1]})
            # pref.push().set({"latitude": speedbreakers[i][0], "longitude": speedbreakers[i][1], "label": "Speedbreaker"})
            # sref.push().set({"latitude": speedbreakers[i][0], "longitude": speedbreakers[i][1]})
            folium.CircleMarker(location=[speedbreakers[i][0], speedbreakers[i][1]],
                        radius=6,
                        weight=6, color="green").add_to(this_map)
            
    for i in range(len(blabels)):
        if blabels[i] == -1:
            continue
        elif blabels[i] not in bunique:
            bunique.add(blabels[i])
            # final_b.append({"latitude": badroads[i][0], "longitude": badroads[i][1]})
            # pref.push().set({"latitude":  badroads[i][0], "longitude": badroads[i][1], "label": "Bad road"})
            # bref.push().set({"latitude": badroads[i][0], "longitude": badroads[i][1]})
            folium.CircleMarker(location=[badroads[i][0], badroads[i][1]],
                        radius=6,
                        weight=6, color="blue").add_to(this_map)

    for i in range(len(plabels)):
        if plabels[i] == -1:
            continue
        elif plabels[i] not in punique:
            punique.add(plabels[i])
            # final_p.append({"latitude": potholes[i][0], "longitude": potholes[i][1]})
            # pref.push().set({"latitude": potholes[i][0], "longitude": potholes[i][1]})
            folium.CircleMarker(location=[potholes[i][0], potholes[i][1]],
                        radius=6,
                        weight=6, color="red").add_to(this_map)
        

    #Set the zoom to the maximum possible
    this_map.fit_bounds(this_map.get_bounds())

    #Save the map to an HTML file
    this_map.save('folium_visualization_new.html')

# merge()

def visualize(locs):

    potholes = locs[0]
    badroads = locs[1]
    speedbreakers = locs[2]

    this_map = folium.Map(prefer_canvas=True)

    for i in range(len(speedbreakers)):
        folium.CircleMarker(location=[speedbreakers[i][0], speedbreakers[i][1]],
                    radius=6,
                    weight=6, color="green").add_to(this_map)
            
    for i in range(len(badroads)):
        folium.CircleMarker(location=[badroads[i][0], badroads[i][1]],
                    radius=6,
                    weight=6, color="blue").add_to(this_map)

    for i in range(len(potholes)):
        folium.CircleMarker(location=[potholes[i][0], potholes[i][1]],
                    radius=6,
                    weight=6, color="red").add_to(this_map)
        

    #Set the zoom to the maximum possible
    this_map.fit_bounds(this_map.get_bounds())

    #Save the map to an HTML file
    this_map.save('unlabelled.html')

# visualize(new_locs())
new_locs()