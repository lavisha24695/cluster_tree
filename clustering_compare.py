import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

np.random.seed(0)


# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 2000
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,noise=.05)
# file = open('noisy_circles.txt','w')
# for line in noisy_circles[0]:
#         file.write(" ".join(str(elem) for elem in line) + "\n")
# file.close
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
# file = open('noisy_moons.txt','w')
# for line in noisy_moons[0]:
#         file.write(" ".join(str(elem) for elem in line) + "\n")
# file.close
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
# file = open('blobs.txt','w')
# for line in blobs[0]:
#         file.write(" ".join(str(elem) for elem in line) + "\n")
# file.close
no_structure = np.random.rand(n_samples, 2), None
file = open('no_structure.txt','w')
for line in no_structure[0]:
        file.write(" ".join(str(elem) for elem in line) + "\n")
file.close
# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)
# file = open('aniso.txt','w')
# for line in aniso[0]:
#         file.write(" ".join(str(elem) for elem in line) + "\n")
# file.close
# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples,cluster_std=[1.0, 2.5, 0.5],random_state=random_state)
# file = open('varied.txt','w')
# for line in varied[0]:
#         file.write(" ".join(str(elem) for elem in line) + "\n")
# file.close
#############
# data_handle = open('/Users/lavisha/PycharmProjects/Project1/data_aggregation.txt', "r")
# data1 = data_handle.read().split('\n')[:-1]
# data = [eg.split()[:2] for eg in data1]
# data = [[float(feature) for feature in example] for example in data]
# y = [eg.split()[2] for eg in data1]
# y = [[float(feature) for feature in example]for example in y]
# mydataset1 = [np.array(data),np.array(y)]
# ##################
# data_handle = open('/Users/lavisha/PycharmProjects/Project1/data_crescents.txt', "r")
# data1 = data_handle.read().split('\n')[:-1]
# data = [eg.split()[:2] for eg in data1]
# data = [[float(feature) for feature in example] for example in data]
# # y = [eg.split()[2] for eg in data1]
# # y = [[float(feature) for feature in example]for example in y]
# y = np.ones(len(data))
# mydataset2 = [np.array(data),np.array(y)]
#
#
# # ============
# # Set up cluster parameters
# # ============
# plt.figure(figsize=(4 * 2 + 3, 12.5))
# plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
#                     hspace=.01)
#
# plot_num = 1
#
# default_base = {'quantile': .3,
#                 'eps': .3,
#                 'damping': .9,
#                 'preference': -200,
#                 'n_neighbors': 10,
#                 'n_clusters': 3}
#
# datasets = [
#     (noisy_circles, {'damping': .77, 'preference': -240,'quantile': .2, 'n_clusters': 2}),
#     (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
#     (varied, {'eps': .18, 'n_neighbors': 2}),
#     (aniso, {'eps': .15, 'n_neighbors': 2}),
#     (blobs, {}),
#     (no_structure, {}),
#     (mydataset1, {}),
#     (mydataset2, {})]
#
# for i_dataset, (dataset, algo_params) in enumerate(datasets):
#     # update parameters with dataset-specific values
#     params = default_base.copy()
#     params.update(algo_params)
#
#     X, y = dataset
#
#     # normalize dataset for easier parameter selection
#     X = StandardScaler().fit_transform(X)
#
#     # estimate bandwidth for mean shift
#     # bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])
#
#     # connectivity matrix for structured Ward
#     connectivity = kneighbors_graph(X, n_neighbors=params['n_neighbors'], include_self=False)
#     # make connectivity symmetric
#     connectivity = 0.5 * (connectivity + connectivity.T)
#
#     # ============
#     # Create cluster objects
#     # ============
#     # ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
#     # two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
#     # ward = cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='ward', connectivity=connectivity)
#     spectral = cluster.SpectralClustering(n_clusters=params['n_clusters'], eigen_solver='arpack',affinity="nearest_neighbors")
#     dbscan = cluster.DBSCAN(eps=params['eps'])
#     # affinity_propagation = cluster.AffinityPropagation(damping=params['damping'], preference=params['preference'])
#     average_linkage = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock",n_clusters=params['n_clusters'], connectivity=connectivity)
#     # birch = cluster.Birch(n_clusters=params['n_clusters'])
#     gmm = mixture.GaussianMixture(n_components=params['n_clusters'], covariance_type='full')
#     kmeans = KMeans(n_clusters=params['n_clusters'], random_state=0)
#
#     clustering_algorithms = (
#         # ('MiniBatchKMeans', two_means),
#         # ('AffinityPropagation', affinity_propagation),
#         # ('MeanShift', ms),
#         ('SpectralClustering', spectral),
#         # ('Ward', ward),
#         ('AgglomerativeClustering', average_linkage),
#         ('DBSCAN', dbscan),
#         # ('Birch', birch),
#         ('GaussianMixture', gmm),
#         ('K-Means', kmeans)
#     )
#
#     for name, algorithm in clustering_algorithms:
#         t0 = time.time()
#
#         # catch warnings related to kneighbors_graph
#         with warnings.catch_warnings():
#             warnings.filterwarnings(
#                 "ignore",
#                 message="the number of connected components of the " +
#                 "connectivity matrix is [0-9]{1,2}" +
#                 " > 1. Completing it to avoid stopping the tree early.",
#                 category=UserWarning)
#             warnings.filterwarnings(
#                 "ignore",
#                 message="Graph is not fully connected, spectral embedding" +
#                 " may not work as expected.",
#                 category=UserWarning)
#             algorithm.fit(X)
#
#         t1 = time.time()
#
#         if hasattr(algorithm, 'labels_'):
#             y_pred = algorithm.labels_.astype(np.int)
#         else:
#             y_pred = algorithm.predict(X)
#
#         plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
#         if i_dataset == 0:
#             plt.title(name, size=18)
#
#         colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
#                                              '#f781bf', '#a65628', '#984ea3',
#                                              '#999999', '#e41a1c', '#dede00']),
#                                       int(max(y_pred) + 1))))
#         # add black color for outliers (if any)
#         colors = np.append(colors, ["#000000"])
#         plt.scatter(X[:, 0], X[:, 1], s = 1, color=colors[y_pred])
#
#         plt.xlim(-2.5, 2.5)
#         plt.ylim(-2.5, 2.5)
#         plt.xticks(())
#         plt.yticks(())
#         plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
#                  transform=plt.gca().transAxes, size=15,
#                  horizontalalignment='right')
#         plot_num += 1
#
# plt.show()