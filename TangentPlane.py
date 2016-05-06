from __future__ import division

import numpy as np
import scipy as sp
import math

from sklearn.decomposition import PCA

class TangentPlane(object):
    """
        TangentPlane is the data structure for representing a tangent plane
    """

    def __init__(self, point, dimension, neighbors=None, basis=None):
        self._point = point
        self._dimension = dimension
        self._basis = None
        self._neighbors = None

        self._pca = PCA(n_components=self._dimension)

        if(neighbors is None):
            self._basis = basis
        else:
            self._neighbors = neighbors
            self.compute_basis()

    @property
    def data_manifold(self): return self._point

    @property
    def dimension(self): return self._dimension

    @property
    def basis(self): return self._basis

    @property
    def neighbors(self): return self._neighbors

    def compute_basis(self):
        self._pca.fit(self.neighbors)
        self._basis = self._pca.components_
