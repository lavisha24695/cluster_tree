import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from mpl_toolkits.mplot3d import Axes3D

dataset = '/Users/lavisha/PycharmProjects/Project1/data_aggregation.txt'
# dataset = '/Users/lavisha/PycharmProjects/Project1/data_crescents.txt'
# dataset = '/Users/lavisha/PycharmProjects/Project1/noisy_circles.txt'
# dataset = '/Users/lavisha/PycharmProjects/Project1/noisy_moons.txt'
# dataset = '/Users/lavisha/PycharmProjects/Project1/blobs.txt'
# dataset = '/Users/lavisha/PycharmProjects/Project1/aniso.txt'
# dataset = '/Users/lavisha/PycharmProjects/Project1/varied.txt'
# dataset = '/Users/lavisha/PycharmProjects/Project1/no_structure.txt'
# dataset = '/Users/lavisha/PycharmProjects/Project1/my3d.txt'
# dataset = '/Users/lavisha/PycharmProjects/Project1/my3d_analysis_2.txt'
#

d = 2
D = d
# blocksize = 1
branch_factor = 2**d
data_handle = open(dataset, "r")
data = data_handle.read().split('\n')[:-1]
'''Ignore the 3rd component which represents cluster number.'''
data = [eg.split()[:d] for eg in data]
data = [[float(feature) for feature in example]for example in data]#aggregate
data = [[float(feature)*50 for feature in example]for example in data]#varied
'''Scale and Quantize the data '''
data = [[round(f) for f in example] for example in data]
# data = [[example[0]+45, example[1]+17] for example in data]#crescent
# data = [[example[0]+62, example[1]+60 ] for example in data]#noisy circles
# data = [[example[0] + 55, example[1] + 50] for example in data]  # noisy moons
# data = [[example[0] + 80, example[1] + 126] for example in data]  # blobs
# data = [[example[0] + 60, example[1] + 50] for example in data]  # aniso
# data = [[example[0] + 70, example[1] + 50] for example in data]  # varied
# data = [[example[0] + 10, example[1] + 10] for example in data]  # no_Structure
# data = [[example[0] + 35, example[1] + 30, example[2] + 40] for example in data]#3d

# data = [[example[0], example[1]] for example in data if (example[1]**2 + example[0]**2 <1225)]   # varied
# data = [[example[0], example[1]] for example in data if example[1]>32 or example[0]<-9 or (example[0]>15 and example[0]<65) and example[1]>-5]   # varied

print(len(data))
if d == 2:
    print('2d')
    plt.figure()
    plt.plot(np.array(data)[:,0], np.array(data)[:,1], 'k.')
elif d == 3:
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(np.array(data)[:, 0], np.array(data)[:, 1], np.array(data)[:, 2])
    print("max", np.max(np.array(data)[:, 0]), np.max(np.array(data)[:, 1]), np.max(np.array(data)[:, 2]), "min",
          np.min(np.array(data)[:, 0]), np.min(np.array(data)[:, 1]), np.min(np.array(data)[:, 2]))

answer = euclidean_distances(data, data)
print(answer.shape)
all_nos = answer[:]
maxv = round(np.max(all_nos))+1
minv = round(np.min(all_nos))-1
hist, bin_edges = np.histogram(all_nos, bins = np.arange(minv, maxv, 1))
plt.figure()
plt.plot(hist)
plt.show()
