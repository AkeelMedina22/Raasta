import numpy as np
import pickle
import scipy.spatial
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

LOAD_TREE = 1

def load_points(potholes, speedbreakers):

    GOOGLE_APPLICATION_CREDENTIALS="raasta-c542d-firebase-adminsdk-5v3pa-94bf94e3fb.json"

    cred_obj = credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS)
    default_app = firebase_admin.initialize_app(cred_obj, {'databaseURL':"https://raasta-c542d-default-rtdb.asia-southeast1.firebasedatabase.app/"})

    pref = db.reference("/pothole-locations/")
    session_data = pref.get()
    for i in session_data:
        potholes.append([float(session_data[i]['latitude']), float(session_data[i]['longitude'])])

    sref = db.reference("/speedbreaker-locations/")
    session_data = sref.get()
    for i in session_data:
        speedbreakers.append([float(session_data[i]['latitude']), float(session_data[i]['longitude'])])
    
    return potholes, speedbreakers


def build_kd_trees(potholes, speedbreakers):
    p_tree = scipy.spatial.cKDTree(potholes)
    s_tree = scipy.spatial.cKDTree(speedbreakers)
    pickle.dump(p_tree,open('KD_Tree/pothole_tree.p','wb'))
    pickle.dump(s_tree,open('KD_Tree/speedbreaker_tree.p','wb'))


def nearest_neighbor(points):
    p_dist, p_loc, s_dist, s_loc = [],[],[],[]

    with open('KD_Tree/pothole_tree.p', 'rb') as f:
        tree = pickle.load(f)
        p_dist, p_index = tree.query(points)
        p_loc = tree.data[p_index]
    with open('KD_Tree/speedbreaker_tree.p', 'rb') as f:
        tree = pickle.load(f)
        s_dist, s_index = tree.query(points)
        s_loc = tree.data[s_index]

    return p_dist, p_loc, s_dist, s_loc


if __name__ == "__main__":
    points = [[24.5, 67.5], [24.75, 66.5]] # Input from user
    print(nearest_neighbor(points))
