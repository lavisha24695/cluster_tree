import numpy as np
import matplotlib.pyplot as plt
import math
import logging
from random import *
import queue
from PIL import Image
from scipy.misc import imresize
from scipy.spatial import distance
from collections import OrderedDict
from sklearn.neighbors import NearestNeighbors

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # dataset = '/Users/lavisha/PycharmProjects/Project1/data_aggregation.txt'
    dataset = '/Users/lavisha/PycharmProjects/Project1/data_crescents.txt'
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
    data = [[example[0]+45, example[1]+17] for example in data]#crescent
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
    plt.show()
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
    num_exp = 5000
    ct = 1
    xmax = np.max(np.array(data)[:,0])
    ymax = np.max(np.array(data)[:,1])
    print('xmax:', xmax, 'ymax:', ymax)

    # Generate all points possible in the white spaces
    num_points = 0
    my_list = []
    for x1 in range(xmax):
        for y1 in range(ymax):
    # listall = [[59,42],[56,46],[25,13],[48,28],[50,11],[46,45],[32,15],[33,16],[39,42],[55,45],[9,28],[60,43],[39,41],[4,17],[58,9],[49,27],[12,16],[50,9],[16,13],[29,31],[35,34],[36,20],[39,19],[57,27]]
    # listall = [[15,28],[15,29],[15,30], [15,31], [15,32]]
    # for i in range(len(listall)):
    #     # x1 = np.random.randint(1,xmax)
        # y1 = np.random.randint(1,ymax)
        # x1 = listall[i][0]
        # y1 = listall[i][1]
        # for q in range(1):
        # x1 = uniform(1, xmax)
        # y1 = uniform(1, ymax)
        # if ct ==1:
        #     x1,y1 = 18,27
        # elif ct == 2:
        #     x1,y1 = 8,15.5
        # elif ct == 3:
        #     x1,y1 = 27,21.5
            ans = knn.kneighbors([[x1,y1]], 4, return_distance=True)
            datapt = data[ans[1][0][0]]
            if datapt[0] == x1 and datapt[1] == y1:
                print('Skipping')
                continue
            """ 
            # Old technique
            distances = ans[0][0]
            near_pts = [ans[1][0][0]]
            near_dist = [ans[0][0][0]]
            closest_dist = distances[0]
            nbr = 1
            while nbr<len(distances):
                if distances[nbr]<closest_dist+0.1:
                    near_pts.append(ans[1][0][nbr])
                    near_dist.append(ans[0][0][nbr])
                    nbr +=1
                else:
                    break
            nr, dr = 0,0
            for k in range(len(near_pts)):
                nr += near_dist[k]
                dr += meanval[near_pts[k]]

            nr = nr/len(near_pts)
            dr = dr/len(near_pts)
            ratio = nr/dr
            """
            # New technique
            distances = ans[0][0]
            points = ans[1][0]
            # print("traingle check - ", distances[0], distances[1])
            # if distances[0] == np.sqrt(2) and distances[1] == np.sqrt(2):
            #     print("triangle")
            #     continue
            if distances[0] <= np.sqrt(2):
                continue
            if len(distances) != 4 or len(points) != 4:
                print('Wait len not 4')
            nr, dr = 0, 0
            for k in range(4):
                nr += distances[k]
                dr += meanval[points[k]]

            nr = nr/4
            dr = dr/4
            ratio = nr/dr
            if ratio>2.0:
                # For 2D take 2 neighbors and 1 of this point and see if rank of centered matrix is 1 or not
                a1 = np.array([data[ans[1][0][0]], data[ans[1][0][1]], [x1, y1]])
                a2 = a1 - np.mean(a1, axis=0)
                rank1 = np.linalg.matrix_rank(a2)
                if rank1 == 1:
                    print("CLOSE")
                else:
                    continue
            else:
                continue
            # if ratio>2.0:
            #     continue
            # ratio = (ans[0][0][0]/meanval[ans[1][0][0]])
            # print("nr:", nr)
            #
            # if nr == 1.0 and ratio<1.4:
            #     print("Ignoring neighbor")
            #     continue
            pt_info = {'pt': [x1, y1], 'ratio': ratio, 'nr': nr, 'dr': dr, 'next':0, 'nbr1': data[ans[1][0][0]], 'nbr2': data[ans[1][0][1]], 'nbr3': data[ans[1][0][2]], 'nbr4': data[ans[1][0][3]]}
            # if nr == 1.0:
            #     pt_info['next'] = 1
            #     print('Woah.. Found a point unit distance away but with a large ratio')
            # else:
            #     pt_info['next'] = 0
            #
            # print("pt_info", pt_info['pt'])

            my_list.append(pt_info)
            num_points +=1
            # print(pt_info['pt'], pt_info['ratio'], 'nr:', pt_info['nr'], 'dr:', pt_info['dr'], 'next:', pt_info['next'], " ", num_points)
            # plt.figure(ct)
            # plt.plot(data[:, 0], data[:, 1], 'k.')
            # plt.plot(pt_info['pt'][0], pt_info['pt'][1], 'm.')
            # plt.show()


    # """
    print("Number of points:", num_points)
    new_list= sorted(my_list, key=lambda x: x['ratio'])
    ct = 0
    for pt in new_list:
        print(pt['pt'], pt['ratio'], 'nr:', pt['nr'], 'dr:', pt['dr'], 'next:', pt['next'], " ", ct, 'nbr1:', pt['nbr1'], pt['nbr2'], pt['nbr3'], pt['nbr4'])
        # if ratio<1.5 or ratio>2.5:
        #     print('Obvious')
        #     continue
        # oldratio = ratio*1.25
        # print(ct, 'Ratio: %.3f ' % ratio, 'New pt:', [x1, y1],' Dist:', ans[0][0][0], ' Density: %.3f ' % meanval[ans[1][0][0]],  ' Closest pt: ',datapt)
        # print(ct, 'Ratio: %.3f ' % ratio, 'No:', len(near_pts) , near_dist, 'New pt:', [x1, y1], ' Dist:', nr,' Density: %.3f ' % dr, ' Closest pt: ', datapt)

        plt.figure(ct)
        # plt.subplot(1,2,1)
        plt.plot(data[:, 0], data[:, 1], 'k.')
        plt.plot(pt['pt'][0], pt['pt'][1], color='darkviolet', marker='.')
        # plt.subplot(1,2,2)
        # plt.plot(data[:, 0], data[:, 1], 'k.')
        # plt.plot(datapt[0], datapt[1], 'y.')
        ct += 1
        plt.show()
    # """