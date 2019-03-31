import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import csv

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline



WAYPOINTS_FILENAME = 'racetrack_waypoints.txt'  # waypoint file to load

#############################################
# Load Waypoints
#############################################
# Opens the waypoint file and stores it to "waypoints"
waypoints_file = WAYPOINTS_FILENAME
waypoints_np   = None
with open(waypoints_file) as waypoints_file_handle:
    waypoints = list(csv.reader(waypoints_file_handle, 
                                delimiter=',',
                                quoting=csv.QUOTE_NONNUMERIC))
    waypoints_np = np.array(waypoints)


Pts3D = np.squeeze(waypoints_np)
x = Pts3D[:,0]
y = Pts3D[:,1]
z = Pts3D[:,2]

Pts2D = Pts3D[:,[0,1]]

# create matrix versions of these arrays
X = x[:, np.newaxis]
Y = y[:, np.newaxis]

colors = ['teal', 'yellowgreen', 'gold']
lw = 1
# plt.plot(x, y, z, projection='3d', color='cornflowerblue', linewidth=lw,
#          label="ground truth")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#ax.scatter(x, y, z, color='navy', s=30, marker='o', label="training points")
ax.plot(x, y, z, color='red', linewidth=lw, label="ground truth")

for count, degree in enumerate([3,15]):
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(Pts2D,z)
    z_plot = model.predict(Pts2D)
    ax.plot(X, Y, z_plot, color=colors[count], linewidth=lw,
             label="degree %d" % degree)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Speed')

plt.legend(loc='lower left')

plt.show()








# poly = PolynomialFeatures(degree=3)
# X_t = poly.fit_transform(Pts3D)



# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')



# ax.scatter(x, y, z, c='r', marker='o')
# #ax.scatter(x, y, z, c='r', marker='o')

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Speed')



# plt.show()
# # clf = LinearRegression()
# # clf.fit(X_t, y)
# # print(clf.coef_)
# # print(clf.intercept_)