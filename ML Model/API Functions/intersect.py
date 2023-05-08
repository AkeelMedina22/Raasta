import firebase_admin
import numpy as np
from firebase_admin import credentials
from firebase_admin import db
import folium

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

def intersection(points, potholes, speedbreakers):

   def is_between(a,b,c):
      EPSILON = 0.0001
      return np.linalg.norm(a-c) + np.linalg.norm(b-c) - np.linalg.norm(a-b) <= EPSILON


   def data_on_route(points, potholes, speedbreakers):
      '''
      points = [[[float, float], [float, float]], ...]
      '''
      on_route = []
      for i in range(len(points)-1):
         for j in range(len(potholes)):
            if is_between(np.array(points[i]), np.array(points[i+1]), np.array(potholes[j])):
               on_route.append(potholes[j])
         for j in range(len(speedbreakers)):
            if is_between(np.array(points[i]), np.array(points[i+1]), np.array(speedbreakers[j])):
               on_route.append(speedbreakers[j])

      return on_route
   
   intersect = data_on_route(points, potholes, speedbreakers)
   result = list(set(tuple(x) for x in intersect))

   print("There are {} potholes on the route".format(len(result)))
   return result


potholes, speedbreakers = load_points(potholes=[], speedbreakers=[])
points = [(24.91098, 67.12646), (24.91111, 67.12629), (24.91123, 67.12641), (24.91147, 67.12615), (24.91165, 67.12598), (24.91226, 67.12545), (24.91356, 67.1243), (24.91377, 67.12411), (24.91384, 67.1242), (24.91285, 67.1251), (24.91225, 67.12566), (24.91171, 67.12611), (24.91143, 67.12637), (24.91124, 67.12659), (24.91134, 67.12667), (24.91157, 67.12694), (24.91181, 67.1272), (24.91232, 67.12769), (24.91251, 67.1279), (24.91325, 67.12869), (24.91351, 67.12897), (24.91378, 67.12867), (24.91422, 67.12818), (24.91525, 67.1293), (24.91591, 67.12996), (24.91632, 67.13036), (24.91573, 67.13086), (24.91549, 67.13109), (24.91489, 67.13162), (24.91417, 67.13222), (24.91179, 67.13428), (24.90927, 67.13644), (24.90876, 67.13687), (24.90864, 67.13698), (24.90855, 67.13708), (24.90836, 67.1375), (24.90801, 67.13831), (24.90762, 67.13921), (24.90826, 67.13944), (24.90821, 67.13957), (24.90786, 67.13943), (24.90689, 67.13912), (24.90634, 67.13898), (24.90604, 67.13895), (24.90603, 67.13885), (24.90606, 67.13874), (24.90611, 67.13862), (24.90609, 67.13851), (24.90614,67.13835)]
onn_route = intersection(points, potholes, speedbreakers)