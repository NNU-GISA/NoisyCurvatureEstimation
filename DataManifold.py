from __future__ import division

import numpy as np
import scipy as sp
import math

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

class DataManifold(object):
    """
        The DataManifold class contains all of the structure needed to analyze
        an arbitrary point cloud from a manifold theoretic perspective.
    """

    def __init__(self, data_frame, num_neighbors=10):
        self._data_frame = data_frame

        self.parameters = type('', (), {})()
        self.parameters._num_neighbors = num_neighbors

        self._num_points = self._data_frame.shape[0]
        self._neighbors = None
        self._distances = None
        self._adjacency_graph = None
        self._knn_graph = None

        self._compute_neighbors()
        self._compute_distances()

    @property
    def num_neighbors(self): return self.parameters._num_neighbors #see if there is a way to make it parameters.num_neighbors or use dict

    @property
    def data_frame(self): return self._data_frame

    @property
    def neighbors(self): return self._neighbors

    @property
    def distances(self): return self._distances

    @property
    def adjacency_graph(self): return self._adjacency_graph

    @data_frame.setter
    def data_frame(self,value):
        self._data_frame = value

        self._compute_neighbors()
        self._compute_distances()

    @num_neighbors.setter
    def num_neighbors(self,value):
        self.parameters._num_neighbors = value

        self._compute_neighbors()
        self._compute_distances()

    def _compute_neighbors(self):
        V,dim = self.data_frame.shape
        neighbors = NearestNeighbors(n_neighbors=self.num_neighbors,algorithm='auto').fit(self.data_frame)
        _,indices = neighbors.kneighbors(self.data_frame)
        self._adjacency_graph = neighbors.kneighbors_graph(self.data_frame,mode='connectivity')
        self._knn_graph = neighbors.kneighbors_graph(self.data_frame,mode='distance')
        self._neighbors = indices

    def _compute_distances(self):
        self._distances = sp.spatial.distance.squareform(sp.spatial.distance.pdist(self.data_frame,'euclidean'))
