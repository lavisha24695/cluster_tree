import numpy as np
import matplotlib.pyplot as plt
import math
import logging
import queue
from PIL import Image
from scipy.misc import imresize
from scipy.spatial import distance
from collections import OrderedDict
from sklearn.neighbors import NearestNeighbors

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    dataset = '/Users/lavisha/PycharmProjects/Project1/data_aggregation.txt'
    # dataset = '/Users/lavisha/PycharmProjects/Project1/data_crescents.txt'
    # dataset = '/Users/lavisha/PycharmProjects/Project1/blobs.txt'
    d = 2
    D = d
    blocksize = 1
    branch_factor = 2**d
    data_handle = open(dataset, "r")
    data = data_handle.read().split('\n')[:-1]
    '''Ignore the 3rd component which represents cluster number.'''
    data = [eg.split()[:d] for eg in data]
    data = [[float(feature) for feature in example]for example in data]
    # '''Scale and Quantize the data '''
    data = [[round(f) for f in example] for example in data]
    # data = [[example[0]+45, example[1]+17] for example in data]#crescent
    # data = [[example[0] + 80, example[1] + 126] for example in data]  # blobs
    # data = [[2,2],[4,5],[7,8],[1,6],[7,3],[5,5],[8,5],[3,3],[4,6],[7,7]]
    val = 0
    for example in data:
        if max(example)>val:
            val = max(example)
    print(val)
    L = math.ceil(math.log2(val))
    logger.info('n = %d d = %d val = %d L = %d',len(data), len(data[0]), val, L )

    data = np.array(data)
    data = np.unique(data, axis=0)
    if d == 2:
        print('2d')
        plt.figure()
        plt.plot(np.array(data)[:,0], np.array(data)[:,1], 'k.')

    # dist = distance.cdist(data, data, 'euclidean')
    # print(dist)


    knn = NearestNeighbors(n_neighbors=5).fit(data)
    distances, indices = knn.kneighbors(data)

    meanval = np.mean(distances, axis = 1)
    print(meanval)
    # sample = 100
    # print(indices[sample], distances[sample])
    # for nbr in indices[sample]:
    #     plt.plot(data[nbr][0], data[nbr][1], 'r.')
    # plt.plot(data[sample][0], data[sample][1], 'b.')

    # plt.figure()
    # plt.hist(meanval)
    # plt.show()
    num_exp = 90
    ct = 1
    for i in range(num_exp):
        x1 = np.random.randint(1,30)
        y1 = np.random.randint(1,30)
        ans = knn.kneighbors([[x1,y1]], 1, return_distance=True)
        datapt = data[ans[1][0][0]]
        if datapt[0] == x1 and datapt[1] == y1:
            print('Skipping')
            continue
        ratio = (ans[0][0][0]/meanval[ans[1][0][0]])
        print(ct, 'Ratio: %.3f ' % ratio, 'New pt:', [x1, y1],' Dist:', ans[0][0][0], ' Density: %.3f ' % meanval[ans[1][0][0]],  ' Closest pt: ',datapt)

        plt.figure(ct)
        plt.plot(data[:, 0], data[:, 1], 'k.')
        plt.plot(x1,y1, 'r.')
        # plt.plot(datapt[0], datapt[1], 'y.')
        ct += 1
        plt.show()
