from reconstruct import get_locs
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import folium
import firebase_admin
import numpy as np
from firebase_admin import credentials
from firebase_admin import db

GOOGLE_APPLICATION_CREDENTIALS="raasta-c542d-firebase-adminsdk-5v3pa-94bf94e3fb.json"

cred_obj = credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS)
default_app = firebase_admin.initialize_app(cred_obj, {'databaseURL':"https://raasta-c542d-default-rtdb.asia-southeast1.firebasedatabase.app/"})

pref = db.reference("/pothole-locations/")
bref = db.reference("/badroad-locations/")
sref = db.reference("/speedbreaker-locations/")

data = get_locs()

potholes = data[1]
badroads = data[2]
speedbreakers = data[3]
pclusters = DBSCAN(eps=0.0005, min_samples=2, p=2).fit(potholes)
plabels = pclusters.labels_

bclusters = DBSCAN(eps=0.0005, min_samples=2, p=2).fit(badroads)
blabels = bclusters.labels_

sclusters = DBSCAN(eps=0.0005, min_samples=2, p=2).fit(speedbreakers)
slabels = sclusters.labels_

punique = set()
bunique = set()
sunique = set()

this_map = folium.Map(prefer_canvas=True)

for i in range(len(slabels)):
    if slabels[i] == -1:
        continue
    elif slabels[i] not in sunique:
        sunique.add(slabels[i])
        # pref.push().set({"latitude": speedbreakers[i][0], "longitude": speedbreakers[i][1], "label": "Speedbreaker"})
        sref.push().set({"latitude": speedbreakers[i][0], "longitude": speedbreakers[i][1]})
        folium.CircleMarker(location=[speedbreakers[i][0], speedbreakers[i][1]],
                    radius=6,
                    weight=6, color="green").add_to(this_map)
        
for i in range(len(blabels)):
    if blabels[i] == -1:
        continue
    elif blabels[i] not in bunique:
        bunique.add(blabels[i])
        # pref.push().set({"latitude":  badroads[i][0], "longitude": badroads[i][1], "label": "Bad road"})
        bref.push().set({"latitude": badroads[i][0], "longitude": badroads[i][1]})
        folium.CircleMarker(location=[badroads[i][0], badroads[i][1]],
                    radius=6,
                    weight=6, color="blue").add_to(this_map)

for i in range(len(plabels)):
    if plabels[i] == -1:
        continue
    elif plabels[i] not in punique:
        punique.add(plabels[i])
        # pref.push().set({"latitude": potholes[i][0], "longitude": potholes[i][1], "label": "Pothole"})
        pref.push().set({"latitude": potholes[i][0], "longitude": potholes[i][1]})
        folium.CircleMarker(location=[potholes[i][0], potholes[i][1]],
                    radius=6,
                    weight=6, color="red").add_to(this_map)
    
#Set the zoom to the maximum possible
this_map.fit_bounds(this_map.get_bounds())

#Save the map to an HTML file
this_map.save('folium_visualization.html')
print(len(potholes), len(badroads), len(speedbreakers))