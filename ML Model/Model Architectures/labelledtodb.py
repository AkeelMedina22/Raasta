import firebase_admin
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import itertools
from firebase_admin import credentials
from firebase_admin import db

def save_to_db():

    GOOGLE_APPLICATION_CREDENTIALS="raasta-c542d-firebase-adminsdk-5v3pa-94bf94e3fb.json"

    cred_obj = credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS)
    default_app = firebase_admin.initialize_app(cred_obj, {'databaseURL':"https://raasta-c542d-default-rtdb.asia-southeast1.firebasedatabase.app/"})

    ref = db.reference("/sensor-data/")
    session_data = list(ref.get().values())


    latitude = []
    longitude = []
    labels = []

    for session in session_data:

        for key in sorted(session):
            
            try:
                labels.append(session[key]['label'])
                latitude.append(float(session[key]['latitude']))
                longitude.append(float(session[key]['longitude']))
            except:
                pass
    
    ref1 = db.reference("/pothole-locations/")

    for i in range(len(labels)):
        print(i)
        if labels[i] == "Pothole":
            ref1.push().set({"latitude": latitude[i], "longitude": longitude[i], "label": "Pothole"})
        elif labels[i] == "Bad Road":
            ref1.push().set({"latitude": latitude[i], "longitude": longitude[i], "label": "Bad Road"})
        elif labels[i] == "Speedbreaker":
            ref1.push().set({"latitude": latitude[i], "longitude": longitude[i], "label": "Speedbreaker"})
            
save_to_db()