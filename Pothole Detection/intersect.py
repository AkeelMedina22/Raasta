import math
import firebase_admin
import numpy as np
from firebase_admin import credentials
from firebase_admin import db

def get_data():

   GOOGLE_APPLICATION_CREDENTIALS="raasta-c542d-firebase-adminsdk-5v3pa-94bf94e3fb.json"

   cred_obj = credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS)
   default_app = firebase_admin.initialize_app(cred_obj, {'databaseURL':"https://raasta-c542d-default-rtdb.asia-southeast1.firebasedatabase.app/"})

   ref = db.reference("/pothole-locations/")
   session_data = list(ref.get().values())

   potholes = []

   for session in session_data:

      try:
            potholes.append([session['latitude'], session['longitude']])
      except:
            pass
   
   return potholes


def is_between(a,b,c):

   EPSILON = 0.001
   def dot(v,w): return v[0]*w[0] + v[1]*w[1]
   def wedge(v,w): return v[0]*w[1] - v[1]*w[0]

   v = a - b
   w = b - c

   return math.isclose(wedge(v,w), 0.0, abs_tol=EPSILON) and math.isclose(dot(v,w), 0.0, abs_tol=EPSILON)


def potholes_on_route(points):
   '''
   points = [[[float, float], [float, float]], ...]
   '''
   on_route = []
   potholes = get_data()
   for i in range(len(points)):
      for j in range(len(potholes)):
         if is_between(points[i][0], points[i][1], potholes[j]):
            on_route.append(potholes[j])

   return on_route


point1 = np.array([24.886820, 67.144712])
point2 = np.array([24.887132, 67.130851])
result = potholes_on_route([[point1, point2]])

print("There are {} potholes on the route at {} these locations".format(len(result), result))