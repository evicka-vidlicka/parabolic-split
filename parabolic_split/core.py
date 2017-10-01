"""
Created on Fri Sep  8 14:41:37 2017

@author: vidlicko
"""
import math
import numpy as np
from lazy import lazy
# import matplotlib.pyplot as plt


class ProblemModel(object):
    def __init__(self, dimensions, interval, n_of_spatial_points, dt, n_of_steps, rank):
        # number of dimensions
        self.N_OF_DIMENSIONS = dimensions
        # number of collocation points in each dimension
        self.N_OF_COLLOC_POINTS_PER_DIM = 2 * np.ones(self.N_OF_DIMENSIONS, dtype = int)
        # total number of collocation points
        self.N_OF_COLLOC_POINTS = int(
            self.N_OF_COLLOC_POINTS_PER_DIM[0] \
            * self.N_OF_COLLOC_POINTS_PER_DIM[1]
        )
        # interval in spatial variables
        self.INTERVAL = interval
        # number of discretization points for spatial variables
        self.N_OF_SPATIAL_POINTS = n_of_spatial_points
        # time discretization values
        self.DT = dt
        self.N_OF_STEPS = n_of_steps
        # number of dimensions for approximation
        self.RANK = rank

    @lazy
    def discretization_points(self):
        """
        Some description comes here.
        """
        return np.linspace(*self.INTERVAL, self.N_OF_SPATIAL_POINTS)

    @lazy
    def collocation_points(self):
        """
        Some description comes here.
        """
        xv, yv = np.meshgrid(np.linspace(
            -math.sqrt(3),
            math.sqrt(3),
            self.N_OF_COLLOC_POINTS_PER_DIM[0]),
            np.linspace(
                -math.sqrt(6),
                math.sqrt(6),
                self.N_OF_COLLOC_POINTS_PER_DIM[1]
            )
        )

        points = np.zeros((2, self.N_OF_COLLOC_POINTS))

        k = 0
        for i in range(self.N_OF_COLLOC_POINTS_PER_DIM[0]):
            for j in range(self.N_OF_COLLOC_POINTS_PER_DIM[1]):
                points[0, k] = xv[i,j]
                points[1, k] = yv[i,j]
                k += 1

        return points

    @lazy
    def solution():
        pass

    def reON(self, matrixU, matrixA, matrixY):
        U0, A0 = matrixU, matrixA
        tmp1 = math.sqrt(integral(matrixU[:,0] * matrixU[:,0], self.INTERVAL))
        matrixU *= 1./tmp1
        tmp2 = math.sqrt(mean_value(matrixY[0,:] * matrixY[0,:]))
        matrixY *= 1./tmp2
        matrixA *= tmp1 * tmp2

        Y_mean = np.zeros((self.RANK, np.shape(matrixY)[1]))
        for i in range(self.RANK):
            Y_mean[i,:] = mean_value(matrixY[i,:]) * np.ones(self.N_OF_COLLOC_POINTS)
        matrixY -= Y_mean
        u_mean = np.dot(np.dot(U0, A0), Y_mean)

        return u_mean, matrixU, matrixA, matrixY

    def x(self, matrixU):
        a, b = self.INTERVAL
        U = np.zeros((np.shape(matrixU)[0] + 2, np.shape(matrixU)[1]))
        U[1:-1,:] = matrixU
        Ux = np.zeros((np.shape(matrixU)[0] + 2, np.shape(matrixU)[1]))
        for i in range(np.shape(matrixU)[1]):
            Ux[:,i] = np.gradient(U[:,i], (b-a)/self.N_OF_STEPS)
        return Ux[1:-1,:]

    def xx(self, matrixU):
        a, b = self.INTERVAL
        U = np.zeros((np.shape(matrixU)[0] + 2, np.shape(matrixU)[1]))
        U[1:-1,:] = matrixU
        Ux = np.zeros((np.shape(matrixU)[0] + 2, np.shape(matrixU)[1]))
        Uxx = np.zeros((np.shape(matrixU)[0] + 2, np.shape(matrixU)[1]))
        for i in range(np.shape(matrixU)[1]):
            Ux[:,i] = np.gradient(U[:,i], (b-a)/self.N_OF_STEPS)
            Uxx[:,i] = np.gradient(Ux[:,i], (b-a)/self.N_OF_STEPS)
        return Uxx[1:-1,:]




def mean_value(vector: np.array):
    """
    Computes the mean value of a vector.

    :param vector:
    :return: A real number.
    """
    print(type(vector))
    return (1./np.shape(vector)[0]) * np.sum(vector)


def integral(vector: np.array, interval):
    """
    Computes an integral of a given vector through an interval.

    :param vector:
    :param interval: Pair of real numbers determinig the interval through which to integrate.

    :return: A real number.
    """
    a, b = interval
    return (b - a) * mean_value(vector)




def L(matrixU, matrixA, matrixY, rank):
    U = np.zeros((np.shape(matrixU)[0] + 2, rank))
    U[1:-1,:] = matrixU
    Ux = np.ndarray((np.shape(matrixU)[0] + 2, rank))
    Uxx = np.ndarray((np.shape(matrixU)[0] + 2, rank))
    for i in range(np.shape(matrixU)[1]):
        Ux[:,i] = np.gradient(U[:,i], (b-a)/N_x)
        Uxx[:,i] = np.gradient(Ux[:,i], (b-a)/N_x)
    #print "check per partes error: " abs()
    return Uxx[1:-1, :], Ux[1:-1, :], matrixA, matrixY
