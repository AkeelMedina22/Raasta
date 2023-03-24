import math
import firebase_admin
import numpy as np
from firebase_admin import credentials
from firebase_admin import db
import matplotlib.pyplot as plt
from reconstruct import get_locs

def get_data():

    GOOGLE_APPLICATION_CREDENTIALS = "raasta-c542d-firebase-adminsdk-5v3pa-94bf94e3fb.json"

    cred_obj = credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS)
    default_app = firebase_admin.initialize_app(cred_obj, {
                                                'databaseURL': "https://raasta-c542d-default-rtdb.asia-southeast1.firebasedatabase.app/"})

    ref = db.reference("/pothole-locations/")
    session_data = list(ref.get().values())

    potholes = []

    for session in session_data:

        try:
            potholes.append([session['latitude'], session['longitude']])
        except:
            pass

    return potholes


def is_near(points):

    EPSILON = 0.00001
    unique1 = []
    unique2 = []

    sorted(points, key=lambda x: (x[0]))

    def is_nearr(point0, point1):
        if (point0[0] - point1[0])**2 <= EPSILON and (point0[1] - point1[1])**2 <= EPSILON:
            return True
        return False

    # points[:] = [points[i] for i in range(1, len(points)) if is_near(points[i-1], points[i], unique1)]

    for i in range(1, len(points)):
        if is_nearr(points[i-1], points[i]):
            unique1.append(points[i-1])
            unique1.append(points[i])
            i += 1

    sorted(unique1, key=lambda x: (x[1], x[0]))

    # points[:] = [points[i] for i in range(1, len(points)) if is_near(points[i-1], points[i], unique2)]
    for i in range(1, len(unique1)):
        if is_nearr(unique1[i-1], unique1[i]):
            unique2.append(unique1[i-1])
            unique2.append(unique1[i])
            i += 1

    arr_result = set(tuple(x) for x in unique1)
    arr_result.intersection_update(tuple(x) for x in unique2)
    print(arr_result)
    return arr_result


potholes = get_locs()[1]
result = is_near(potholes)

print("There are {} potholes too near each other".format(len(result)))
plt.scatter([i[0] for i in result], [i[1] for i in result])
plt.show()
