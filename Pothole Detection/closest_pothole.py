import numpy as np
import scipy.spatial

points = np.array([24.8867965, 67.1383555])
x_array = np.array([24.886820, 24.887132])
y_array = np.array([67.144712, 67.130851])

# Shoe-horn existing data for entry into KDTree routines
combined_x_y_arrays = np.dstack([y_array.ravel(),x_array.ravel()])[0]
points_list = list(points.transpose())


def do_kdtree(combined_x_y_arrays,points):
    mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
    dist, indexes = mytree.query(points)
    return indexes

results = do_kdtree(combined_x_y_arrays,points_list)
print(results)