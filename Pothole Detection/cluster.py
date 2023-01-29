import math
import firebase_admin
import numpy as np
from firebase_admin import credentials
from firebase_admin import db
import matplotlib.pyplot as plt


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


potholes = [(24.9470422, 67.0510583), (24.9662395, 67.0671247), (24.9090505, 67.1310046), (24.9398181, 67.0851039), (24.9080753, 67.1333495), (24.9248974, 67.0913802), (24.964248, 67.0666175), (24.9214628, 67.0943681), (24.9440493, 67.0485271), (24.9625, 67.0651394), (24.9573949, 67.0607066), (24.9665487, 67.0670863), (24.9430129, 67.0476841), (24.9224749, 67.0934111), (24.9490369, 67.0528453), (24.9586418, 67.0617771), (24.9481641, 67.0520773), (24.9076795, 67.1342859), (24.9239133, 67.0908941), (24.9467674, 67.0508005), (24.9240448, 67.0914798), (24.9431175, 67.0477476), (24.9112233, 67.1263972), (24.927454, 67.0892003), (24.9564969, 67.0599185), (24.9089971, 67.1311479), (24.9335757, 67.0857762), (24.9104432, 67.1278408), (24.9060668, 67.1180962), (24.9449058, 67.0841016), (24.9252767, 67.0910719), (24.9669287, 67.0670425), (24.9479031, 67.0518421), (24.9188945, 67.0981754), (24.9659395, 67.0671186), (24.9427906, 67.0475197), (24.9232886, 67.0901115), (24.9566436, 67.0600537), (24.9495687, 67.0528328), (24.9279426, 67.0886956), (24.9638791, 67.0662975), (24.936129, 67.0855204), (24.9392003, 67.085161), (24.9187861, 67.0980181), (24.9473235, 67.0513078), (24.9495784, 67.0528141), (24.9582439, 67.0614484), (24.9312925, 67.0862912), (24.9106468, 67.1275456), (24.9495756, 67.0528198), (24.9081331, 67.1332074), (24.9433386, 67.0479153), (24.956077, 67.0595912), (24.9253569, 67.0910132), (24.9463463, 67.0504562), (24.9673593, 67.0669953), (24.9306664, 67.0865452), (24.9567141, 67.0601132), (24.9110373, 67.1267198), (24.9494477, 67.0529676), (24.9110373, 67.1267198), (24.9494477, 67.0529676), (24.9208409, 67.0948792), (24.9484694, 67.05236), (24.962992, 67.0655549), (24.9563385, 67.0598084), (24.9227564, 67.0894128), (24.9434962, 67.0845573), (24.9219758, 67.0938948), (24.9254189, 67.0909648), (24.9239741, 67.0909891), (24.9261102, 67.0904127), (24.9461361, 67.0834863), (24.9104191, 67.1278453)]
result = is_near(potholes)

print("There are {} potholes too near each other".format(len(result)))
plt.scatter([i[0] for i in result], [i[1] for i in result])
plt.show()
