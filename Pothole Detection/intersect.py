import numpy as np
import math

EPSILON = 0.001

pothole = np.array([24.8867965, 67.1383555])
point1 = np.array([24.886820, 67.144712])
point2 = np.array([24.887132, 67.130851])

def dot(v,w): return v[0]*w[0] + v[1]*w[1]
def wedge(v,w): return v[0]*w[1] - v[1]*w[0]

def is_between(a,b,c):
   v = a - b
   w = b - c
   print(dot(v,w))
   print(wedge(v,w))
   return math.isclose(wedge(v,w), 0.0, abs_tol=EPSILON) and math.isclose(dot(v,w), 0.0, abs_tol=EPSILON)

print(is_between(point1, point2, pothole))