import numpy as np
import math

# E = np.array[vehicle_location.x, vehicle_location.y]
# A = np.array[last_waypoint.x, last_waypoint.y]
# B = np.array[next_waypoint.x, next_waypoint.y]

E = np.array([2, 2])
A = np.array([1, 1])
B = np.array([3, 1])

Vector_AE = E - A
Vector_AB = B - A

temp = Vector_AE.dot(Vector_AB)/Vector_AB.dot(Vector_AB)
temp = temp*Vector_AB

distance = np.linalg.norm(Vector_AE - temp)
print(distance)