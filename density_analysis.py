import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math
from sklearn.cluster import KMeans


alldata = []

"""Generate points in a cuboid"""
cube_xsize = 8
cube_ysize = 14
cube_zsize = 7
cube_xoffset = 14
cube_yoffset = -5
cube_zoffset = -5
cube_num = 64
cube_data = []
for i in range(cube_num):
    x = random.uniform(0, cube_xsize) + cube_xoffset
    y = random.uniform(0, cube_ysize) + cube_yoffset
    z = random.uniform(0, cube_zsize) + cube_zoffset
    cube_data.append([x,y,z])

alldata.extend(cube_data)
cube_data = np.array(cube_data)

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter(cube_data[:,0], cube_data[:,1], cube_data[:,2], '.')
# plt.show()


"""Generate points in a cylinder"""
cylinder_length = 16
cylinder_radius = 4
cylinder_xoffset = 5
cylinder_yoffset = 1
cylinder_zoffset = -11
# cylinder_num = 1024
cylinder_num = cube_num
cylinder_data = []
for i in range(cylinder_num):
    r = (random.uniform(0, 1) ** 0.5) * cylinder_radius
    theta = random.uniform(0, 2*np.pi)
    x = r*math.sin(theta) + cylinder_xoffset
    y = r*math.cos(theta) + cylinder_yoffset
    z = random.uniform(0, cylinder_length) + cylinder_zoffset
    cylinder_data.append([x,y,z])
alldata.extend(cylinder_data)
cylinder_data = np.array(cylinder_data)

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter(cylinder_data[:,0], cylinder_data[:,1], cylinder_data[:,2], '.')
# plt.show()


alldata = np.array(alldata)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(alldata[:,0], alldata[:,1], alldata[:,2], '.')
# plt.show()

file = open('my3d_analysis_6.txt','w')
for line in alldata:
        #print line
        file.write(" ".join(str(elem) for elem in line) + "\n")
file.close()

num_clusters = 2
X = alldata
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
Y = kmeans.labels_
cols = []
print('Detected ', num_clusters, ' clusters')
for i in range(num_clusters):
    col = np.random.rand(3, )
    cols.append(col)
cols = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1]]
fig = plt.figure()
ax = plt.axes(projection='3d')
for i in range(len(X)):
    ax.scatter(X[i,0], X[i,1], X[i,2], s=1, c=cols[Y[i]])

# plt.show()