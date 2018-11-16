from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph

num_clusters = 20
# dataset = '/Users/lavisha/PycharmProjects/Project1/data_aggregation.txt'
dataset = '/Users/lavisha/PycharmProjects/Project1/data_crescents.txt'
d = 2
# branch_factor = 2 ** d
data_handle = open(dataset, "r")
data = data_handle.read().split('\n')[:-1]
#Ignore the 3rd component which represents cluster number.
data = [eg.split()[:d] for eg in data]
data = [[float(feature) for feature in example] for example in data]
#Scale and Quantize the data
data = [[round(f) for f in example] for example in data]
data = [[example[0]+45, example[1]+17] for example in data]
X = np.array(data)
####################a
# kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
# Y = kmeans.labels_
#####################
# gmm = mixture.GaussianMixture(n_components=num_clusters, covariance_type='full').fit(X)
# Y = gmm.predict(X)
####################
# dbscan = cluster.DBSCAN(eps=2).fit(X)
# Y = dbscan.labels_
# num_clusters = np.max(Y)+2
#Default min points for DBSCAN is 5
####################
# spectral = cluster.SpectralClustering(n_clusters=num_clusters, eigen_solver='arpack',affinity="nearest_neighbors").fit(X)
# Y = spectral.labels_
#####################
connectivity = kneighbors_graph(X, n_neighbors = 10, include_self=False)
connectivity = 0.5 * (connectivity + connectivity.T)
average_linkage = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock",n_clusters=num_clusters, connectivity=connectivity).fit(X)
Y = average_linkage.labels_
##########################
cols = []
print('Detected ', num_clusters, ' clusters')
for i in range(num_clusters):
    col = np.random.rand(3, )
    cols.append(col)

for i in range(len(X)):
    # print(Y[i]+1)
    plt.scatter(X[i,0], X[i,1], s=9, c=cols[Y[i]])
    # plt.scatter(X[i, 0], X[i, 1], s=3.5, c=cols[Y[i]])

plt.show()