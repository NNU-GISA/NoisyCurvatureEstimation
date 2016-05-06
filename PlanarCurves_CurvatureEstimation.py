from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import math
from matplotlib import cm
import random
from matplotlib.patches import FancyArrowPatch
from DataManifold import DataManifold
from TangentPlane import TangentPlane
from sklearn.decomposition import PCA

def angle(p1, p2):
    theta = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    if theta < 0:
        theta += 2 * math.pi
    return theta

class CurvatureEstimator_Curves(object):
    """
        The CurvatureEstimator_Curves estimates the curvature of planar curves
        in R^2
    """

    def __init__(self, data_manifold):
        self._data_manifold = data_manifold
        self._curvatures = None

    @property
    def data_manifold(self): return self._data_manifold

    @property
    def curvatures(self): return self._curvatures

    def getBoundary_Omega(self, point_idx, r):
        # pretty sure this isn't the best way to do this, but it works for now
        point = self.data_manifold.data_frame[point_idx]
        dists = np.array(self.data_manifold.distances[point_idx,:])
        B = self.data_manifold.data_frame[dists <= r]
        pca = PCA(n_components=1)
        pca.fit(B)
        B_transform = pca.transform(B)

        dB = [B[B_transform.argmin()], B[B_transform.argmax()]]

        theta = np.array([angle(point,dB[0]), angle(point,dB[1])])
        if(theta[0] > theta[1]):
            np.flipud(dB)
            np.flipud(theta)

        return dB, theta

    def getBoundary_Neighborhood(self, neighborhood, r):
        # pretty sure this isn't the best way to do this, but it works for now
        pca = PCA(n_components=1)
        pca.fit(neighborhood)
        B_transform = pca.transform(neighborhood)

        dB = [neighborhood[B_transform.argmin()], neighborhood[B_transform.argmax()]]

        theta = np.array([angle((0,0),dB[0]), angle((0,0),dB[1])])
        if(theta[0] > theta[1]):
            np.flipud(dB)
            np.flipud(theta)

        return dB, theta

    def IntegralOmega(self, f, th, r):
        """
            I(f) = int_{B_r^+} f(x,y) dl - int_{0}^{1/2 kr^2} f(r,y) dy
                     - int_{0}^{1/2 kr^2} f(-r,y) dy (approximately)

            Right now the code will only compute the integral invariants
              necessary since we only need theta_1 and theta_2 so the computations
              are O(1)
            TODO: Write code that uses convolutions to compute arbitrary integrals
                  (I've been having a bit of trouble with this over the past day
                   or so which is the only thing I need to do to have this working
                   for arbitrary hypersurfaces)
        """

        if f == "1":
            L_r = r * (th[1] - th[0])
            return L_r
        elif f == "x^2":
            Ix2 = th[1] - th[0]
            Ix2 += math.sin(th[1]) * math.cos(th[1])
            Ix2 -= math.sin(th[0]) * math.cos(th[0])
            Ix2 *= (r**3) / 2
            return Ix2
        elif f == "y^2":
            Iy2 = th[1] - th[0]
            Iy2 -= math.sin(th[1]) * math.cos(th[1])
            Iy2 += math.sin(th[0]) * math.cos(th[0])
            Iy2 *= (r**3) / 2
            return Iy2
        elif (f == "xy") or (f == "yx"):
            Ixy = math.sin(th[1])**2 - math.sin(th[0])**2
            Ixy *= (r**3) / 2
            return Ixy
        elif f == "x":
            Ix = math.sin(th[1]) - math.sin(th[0])
            Ix *= (r**2)
            return Ix
        elif f == "y":
            Iy = math.cos(th[0]) - math.cos(th[1])
            Iy *= (r**2)
            return Iy
        else:
            return None

    def bootstrap_residuals(self, epsilon, num_points):
        N = np.random.normal(0, 1, num_points)
        residuals = epsilon * (N / math.sqrt(2) + (np.square(N) - 1)/2)
        return residuals

    def bootstrap_estimate(self, point_idx, r):
        point = self.data_manifold.data_frame[point_idx]
        dists = np.array(self.data_manifold.distances[point_idx,:])
        B_p = self.data_manifold.data_frame[dists <= r]
        B_p = B_p - point

        kappa, tangent, normal = self.EstimateCurvature(point_idx, r)

        theta = -angle((0,0),tangent)
        rotMatrix = np.array([[math.cos(theta), -math.sin(theta)],
                                 [math.sin(theta),  math.cos(theta)]])
        neighborhood = np.array(map(lambda p: np.dot(rotMatrix,p), B_p.tolist()))
        epsilon = 0.01
        B = 200
        kappa_estimates = np.zeros(B)
        for b in xrange(B):
            residuals = self.bootstrap_residuals(epsilon=0.01, num_points=np.shape(neighborhood)[0])
            bootstrap_neighborhood = neighborhood
            bootstrap_neighborhood[:,1] += residuals
            kappa_estimates[b] = self.EstimateCurvature_Bootstrap(bootstrap_neighborhood,r)

        std_error = (1/B) * sum(np.square(kappa_estimates - kappa))
        return std_error


    def construct_covariance_matrix(self, th, r):
        L_r = self.IntegralOmega("1", th, r)
        Ix2 = self.IntegralOmega("x^2", th, r)
        Iy2 = self.IntegralOmega("y^2", th, r)
        Ixy = self.IntegralOmega("xy", th, r)

        Ix =  self.IntegralOmega("x", th, r)
        Iy =  self.IntegralOmega("y", th, r)
        barycenter = (1 / L_r) * np.array([[Ix*Ix, Ix*Iy],[Ix*Iy, Iy*Iy]])

        Sigma = np.array([[Ix2, Ixy], [Ixy, Iy2]]) - barycenter

        return Sigma

    def EstimateCurvature_Bootstrap(self, neighborhood, r):
        point = (0,0)
        dB_p, th = self.getBoundary_Neighborhood(neighborhood, r)

        Sigma = self.construct_covariance_matrix(th, r)

        _,s,V = np.linalg.svd(Sigma)
        v1 = V[:,0]
        v2 = V[:,1]

        ab = dB_p[0] - point
        ac = dB_p[1] - point

        C = 0
        tangent = None
        normal = None
        if(math.copysign(1,np.dot(ab,v1)) == math.copysign(1,np.dot(ac,v1))):
            C = (math.pi / (2*r)) - s[1]/(r**4)
            tangent = v2
            normal = v1
        else:
            C = (math.pi / (2*r)) - s[0]/(r**4)
            tangent = v1
            normal = v2
        return C

    def EstimateCurvature(self, point_idx, r):
        point = self.data_manifold.data_frame[point_idx]
        dB_p, th = self.getBoundary_Omega(point_idx, r)

        Sigma = self.construct_covariance_matrix(th, r)

        _,s,V = np.linalg.svd(Sigma)
        v1 = V[:,0]
        v2 = V[:,1]

        ab = dB_p[0] - point
        ac = dB_p[1] - point

        C = 0
        tangent = None
        normal = None
        if(math.copysign(1,np.dot(ab,v1)) == math.copysign(1,np.dot(ac,v1))):
            C = (math.pi / (2*r)) - s[1]/(r**4)
            tangent = v2
            normal = v1
        else:
            C = (math.pi / (2*r)) - s[0]/(r**4)
            tangent = v1
            normal = v2
        return C, tangent, normal

    def ComputeAllCurvature(self,r,bootstrap=None):
        self._curvatures = np.zeros(self.data_manifold._num_points)
        # BOOTSTRAPPING IS NOT WORKING RIGHT NOW !!
        if not(bootstrap is None):
            for i in xrange(self.data_manifold._num_points):
                std_errors = np.zeros(5)
                for j in xrange(5):
                    std_errors[j] = self.bootstrap_estimate(i, (j+1) * 0.2)
                r_optimal = (np.argmin(std_errors) + 1) * 0.2
                self._curvatures[i],_,_ = self.EstimateCurvature(i,r_optimal)
        else:
            for i in xrange(self.data_manifold._num_points):
                self._curvatures[i],_,_ = self.EstimateCurvature(i,r)


if __name__ == "__main__":
    fig = plt.figure(figsize=(6,6))

    ax = fig.add_subplot(111)
    N = 150
    R = 1
    thetas = np.zeros(N)
    x = np.zeros(N)
    y = np.zeros(N)

    eta = 1

    for i in xrange(N):
        thetas[i] = 2 * math.pi * random.random()
    thetas.sort()
    x = thetas
    y = R * np.sin(thetas)
    noise_y = np.random.normal(0,0.05,N)
    #y = y + noise_y

    data_frame = np.c_[x,y]
    M = DataManifold(data_frame=data_frame, num_neighbors=10)
    CurvatureEstimator = CurvatureEstimator_Curves(M)

    r = 0.5
    CurvatureEstimator.ComputeAllCurvature(r)
    true_curvature = np.absolute(np.sin(x)) / np.power((np.cos(x)**2 + np.ones_like(x)), 1.5)

    #ax.scatter(x[1:-1],np.absolute(true_curvature[1:-1]),c='blue',marker='x',s=30)
    ax.scatter(x[1:-1],y[1:-1],c=np.absolute(CurvatureEstimator.curvatures[1:-1]),marker='o',s=30)


    plt.show()
