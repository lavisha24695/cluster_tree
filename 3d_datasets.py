import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math
from sklearn.cluster import KMeans


alldata = []
"""Generate points in a sphere"""
sphere_radius = 4
sphere_xoffset = 1
sphere_yoffset = 2
sphere_zoffset = 3
sphere_num = 400
sphere_data = []
for i in range(sphere_num):
    # r = random.uniform(0, sphere_radius)
    r = (random.uniform(0, 1)**0.33)*sphere_radius
    theta = random.uniform(0,np.pi)
    phi = random.uniform(0, 2*np.pi)
    x = r*math.sin(theta)*math.sin(phi) + sphere_xoffset
    y = r*math.sin(theta)*math.cos(phi) + sphere_yoffset
    z = r*math.cos(theta) + sphere_zoffset
    sphere_data.append([x,y,z])


alldata.extend(sphere_data)
sphere_data = np.array(sphere_data)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter(sphere_data[:,0], sphere_data[:,1], sphere_data[:,2], '.')

"""Generate points in a cuboid"""
cube_xsize = 8
cube_ysize = 4
cube_zsize = 7
cube_xoffset = 11
cube_yoffset = -15
cube_zoffset = -5
cube_num = 400
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
cylinder_length = 22
cylinder_radius = 4
cylinder_xoffset = -12
cylinder_yoffset = 1
cylinder_zoffset = -11
cylinder_num = 400
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

"""Generate points in a circle"""
circle_radius = 2
circle_transformation = np.array([[1,2,1.5],[-2,4,1],[1.9,0,1]])
circle_offset = np.array([-16,0,-23]).reshape((3,1))
circle_num = 400
circle_data = []
for i in range(circle_num):
    r = (random.uniform(0, 1) ** 0.5) * circle_radius
    theta = random.uniform(0, 2 * np.pi)
    x = r * math.sin(theta)
    y = r * math.cos(theta)
    z = 0
    pt = np.array([x,y,z]).reshape((3,1))
    pt1 = np.dot(circle_transformation, pt) + circle_offset
    circle_data.append([pt1[0],pt1[1],pt1[2]])
alldata.extend(circle_data)
circle_data = np.array(circle_data)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter(circle_data[:,0], circle_data[:,1], circle_data[:,2], '.')
# plt.show()


"""Generate points in a rectangle"""
rectangle_xsize = 5
rectangle_ysize = 4
rect_transformation = np.array([[-1,0.3,1],[3,2,0.5],[-1,1,0]])
rectangle_offset = np.array([15,-3,10]).reshape((3,1))
rectangle_num = 400
rectangle_data = []
for i in range(rectangle_num):
    x = random.uniform(0, rectangle_xsize)
    y = random.uniform(0, rectangle_ysize)
    z = 0
    pt = np.array([x,y,z]).reshape((3,1))
    pt1 = np.dot(rect_transformation, pt) + rectangle_offset
    rectangle_data.append([pt1[0],pt1[1],pt1[2]])
alldata.extend(rectangle_data)
rectangle_data = np.array(rectangle_data)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter(rectangle_data[:,0], rectangle_data[:,1], rectangle_data[:,2], '.')
# plt.show()


alldata = np.array(alldata)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(alldata[:,0], alldata[:,1], alldata[:,2], '.')
# plt.show()

file = open('my3d.txt','w')
for line in alldata:
        #print line
        file.write(" ".join(str(elem) for elem in line) + "\n")
file.close


num_clusters = 5
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

plt.show()