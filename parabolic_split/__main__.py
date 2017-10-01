#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 14:41:37 2017

@author: vidlicko
"""

import math
import numpy as np
# import matplotlib.pyplot as plt
from core import *

if __name__ == '__main__':
    # N_col_dim = 2 * np.ones(2, dtype = int)
    # # number of collocation points
    # N_colloc_p = int(N_col_dim[0] * N_col_dim[1])
    # # interval in spatial variables
    # INTERVAL = (0., 2. * math.pi)
    # # number of approx. points in 1 dimension of spatial variables
    # N_x = 5000
    # # time discretization
    # dt = 0.001
    # N_steps = 10

    # the model of the problem.
    model = ProblemModel(
        dimensions=2,
        interval=(0., 2. * math.pi),
        n_of_spatial_points=5000,
        dt=0.001,
        n_of_steps=10,
        rank=2
    )

    # spatial discr. points [0, 2*pi]
    discr_pts = model.discretization_points

    # collocation points for random variables
    colloc_pts = model.collocation_points

    A_corr = np.ndarray((model.N_OF_SPATIAL_POINTS - 2, model.N_OF_COLLOC_POINTS))
    B_corr = np.ndarray((model.N_OF_SPATIAL_POINTS - 2, model.N_OF_COLLOC_POINTS))
    A_corr_xx = np.ndarray((model.N_OF_SPATIAL_POINTS - 2, model.N_OF_COLLOC_POINTS))
    B_corr_xx = np.ndarray((model.N_OF_SPATIAL_POINTS - 2, model.N_OF_COLLOC_POINTS))
    print(colloc_pts[:,0])

    for i in range(model.N_OF_SPATIAL_POINTS - 2):
        for j in range(model.N_OF_COLLOC_POINTS):
            A_corr[i,j] = colloc_pts[0,j] * (1./math.sqrt(math.pi)) * math.sin(discr_pts[i+1])
            B_corr[i,j] = colloc_pts[1,j] * (1./math.sqrt(math.pi)) * math.sin(2*discr_pts[i+1])
            A_corr_xx[i,j] = -colloc_pts[0,j] * (1./math.sqrt(math.pi)) * math.sin(discr_pts[i+1])
            B_corr_xx[i,j] = -4 * colloc_pts[1,j] * (1./math.sqrt(math.pi)) * math.sin(2*discr_pts[i+1])

    # homogenuous boundary condition forced on the approx. solutions (0 first and last )

    #  initial data u0 =
    U0_full = np.zeros((model.N_OF_SPATIAL_POINTS - 2, model.N_OF_COLLOC_POINTS))
    for i in range(model.N_OF_SPATIAL_POINTS - 2):
        for j in range(model.N_OF_COLLOC_POINTS):
            U0_full[i,j] =  colloc_pts[0,j] * (1./math.sqrt(math.pi)) * np.sin(discr_pts[i + 1]) + colloc_pts[1,j] * (1./math.sqrt(math.pi)) * np.sin(2 * discr_pts[i + 1])


    # SVD of initial data
    UU, AA, YY = np.linalg.svd(U0_full, full_matrices = 0, compute_uv = 1)
    print("AA", AA)
    print("error of svd: ", np.linalg.norm(U0_full - np.dot(UU, np.dot(np.diag(AA), YY))))

    # low rank approximation of initial data
    U0 = np.zeros((model.N_OF_SPATIAL_POINTS-2, model.RANK))
    A0 = np.zeros((model.RANK, model.RANK))
    Y0 = np.zeros((model.RANK, model.N_OF_COLLOC_POINTS))

    U0[:, :] = UU[:, :model.RANK]
    A0 = np.diag(AA[:model.RANK])
    Y0[:, :] = YY[:model.RANK, :]


    # reorthonormalization & 0 mean value
    u0_mean, U0, A0, Y0 = model.reON(U0, A0, Y0)


    #print "Frob. norm of u0_mean: ", np.linalg.norm(u0_mean, 'fro')
    print("error of LRA with model.RANK = ",model.RANK,": ", np.linalg.norm(U0_full - np.dot(U0, np.dot(A0, Y0)) , 'fro'))



    u_save = np.zeros(model.N_OF_STEPS)
    error_progress = np.zeros(model.N_OF_STEPS)
    error_U00_progress = np.zeros(model.N_OF_STEPS)
    A00_progress = np.zeros(model.N_OF_STEPS)
    A11_progress = np.zeros(model.N_OF_STEPS)
    error_U00_xx_progress = np.zeros(model.N_OF_STEPS)
    # =========================== MAIN LOOP =======================================
    for step in range(model.N_OF_STEPS):
        print("\n",step)

        # first phase
        Uxx = model.xx(U0)
        #Uxx, Ux, A, Y = L(U0, A0, Y0, model.RANK)
        Ux = model.x(U0)

        t = np.linspace(*model.INTERVAL, model.N_OF_SPATIAL_POINTS-2)
        error_U00_progress[step] = np.linalg.norm(U0[:,0]*A0[0,0]*Y0[0,0] - math.exp(-4*model.DT*(step))*B_corr[:,0])
        error_U00_xx_progress[step] = np.linalg.norm(Uxx[:,0]*A0[0,0]*Y0[0,0] - math.exp(-4*model.DT*(step))*B_corr_xx[:,0])
        matrixTmp = np.zeros((model.RANK, model.RANK))
        for i in range(model.RANK):
            for j in range(model.RANK):
                matrixTmp[i,j] = mean_value(Y0[i,:]*Y0[j,:])

        vectorTmp = np.zeros(model.RANK)
        for i in range(model.RANK):
            vectorTmp[i] = np.abs(integral(Uxx[:,i]*U0[:,i], model.INTERVAL) + integral(Ux[:,i]*Ux[:,i], model.INTERVAL))
        print("error of per partes: ", np.linalg.norm(vectorTmp))


        K1 = np.dot(U0, A0) + model.DT * np.dot(Uxx, np.dot(A0, matrixTmp))
        U1, A1 = np.linalg.qr(K1)
        u_mean, U1_re, A1_re, Y1_re = model.reON(U1, A1, Y0)
        u0_mean += u_mean

        for i in range(model.RANK):
            for j in range(model.RANK):
                matrixTmp[i,j] = integral(U1_re[:,i] * U1_re[:,j], model.INTERVAL)
        print("ON error of U1_re: ", np.linalg.norm(matrixTmp - np.identity(model.RANK)))
        for i in range(model.RANK):
            for j in range(model.RANK):
                matrixTmp[i,j] = mean_value(Y1_re[i,:] * Y1_re[j,:])
        print("ON error of Y1_re: ", np.linalg.norm(matrixTmp - np.identity(model.RANK)))

        print("error of reorthogonalization: ", np.linalg.norm(np.dot(np.dot(U1_re, A1_re), Y1_re) - np.dot(np.dot(U1, A1), Y0), 'fro'))

        # second phase
        matrixTmp2 = np.zeros((model.RANK, model.RANK))
        for i in range(model.RANK):
            for j in range(model.RANK):
                matrixTmp[i,j] = integral(U1_re[:,i]*Uxx[:,j], model.INTERVAL)
                matrixTmp2[i,j] = mean_value(Y0[i,:]*Y1_re[j,:])
        A2 = A1 - model.DT * np.dot(matrixTmp, np.dot(A0,matrixTmp2))
        print("A2", A2)

        # third phase

        K_2 = np.dot(Y1_re.T, A2.T) + model.DT * np.dot(np.dot(Y0.T, A0.T), matrixTmp.T)
        Y3, A3 = np.linalg.qr(K_2)
        Y3 = Y3.T
        A3 = A3.T
        u_mean, U3_re, A3_re, Y3_re = model.reON(U1_re, A3, Y3)
        print("error of reorthogonalization: ", np.linalg.norm(np.dot(np.dot(U3_re, A3_re), Y3_re) - np.dot(np.dot(U1_re, A3), Y3), 'fro'))
        u0_mean += u_mean
        print("Frob norm of u0_mean:", np.linalg.norm(u0_mean, 'fro'))

        print("A3: ", A3_re)

        U0 = U3_re
        A0 = A3_re
        Y0 = Y3_re

        A00_progress[step] = A0[0,0]
        A11_progress[step] = A0[1,1]


        U_corrsol = math.exp(-model.DT*(step+1))*A_corr + math.exp(-4*model.DT*(step+1))*B_corr
        #print "relative error: ", np.linalg.norm(U_corrsol - np.dot(np.dot(U0,A0), Y0), 'fro')/np.linalg.norm(U_corrsol, 'fro')
        error_progress[step] = np.linalg.norm(U_corrsol - np.dot(np.dot(U3_re,A3_re), Y3_re))/np.linalg.norm(U_corrsol)
        # check residual of U_corrsol
        #U_corrsol_xx = _xx(U_corrsol)
        #print "residual of corr sol: ", np.linalg.norm(U_corrsol_xx - U_corrsol_t, 'fro')


        t = np.linspace(*model.INTERVAL, model.N_OF_SPATIAL_POINTS-2)

    t = np.linspace(0., model.DT*model.N_OF_STEPS, model.N_OF_STEPS)
