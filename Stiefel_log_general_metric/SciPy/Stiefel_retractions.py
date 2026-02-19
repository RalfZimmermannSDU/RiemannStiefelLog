#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 15:37:51 2026

@author: zimmermann
"""


import numpy as np
import scipy
from   scipy import linalg
import Stiefel_Exp_Log
import Stiefel_Aux        as StAux
import time

#------------------------------------------------------------------------------
# Polar decomposition retraction
# P_U0 (Xi ) = (U0 + Xi)(In + XiT Xi )^−1/2
# 
# Input arguments      
#          U0    : base point on St(n,p)
#          Xi : tangent vector in T_U0 St(n,p)
# Output arguments
#          U1    : P_X(V)
#------------------------------------------------------------------------------
def Stiefel_PF_ret(U0, Xi):
#------------------------------------------------------------------------------
    # get dimensions
    n,p = U0.shape
    
    # QR decomposition of Xi,
    # only R is needed
    R = np.linalg.qr(Xi, mode='r')
    
    S = np.eye(p,p) + np.dot(R.transpose(), R) #np.dot(Xi.transpose(), Xi);
    # compute matrix square root
    S = linalg.sqrtm(scipy.linalg.inv(S))
        
    # perform U1 = U0*M + Q*N
    U1 = np.dot((U0+Xi),S)
    return U1
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# Polar decomposition inverse retraction
# 
# 
# Input arguments      
#          U0 : base point on St(n,p)
#          U1 :      point on St(n,p)
# Output arguments
#          Xi    : tangent vector in T_U0 St(n,p)
#------------------------------------------------------------------------------
def Stiefel_PF_inv_ret(U0, U1):
#------------------------------------------------------------------------------
    # get dimensions
    n,p = U0.shape
    
    M = (-1)*np.dot(U0.T,U1)
    # solve MX + XM = -2*eye(p)
    
    X = scipy.linalg.solve_sylvester(M, M.T, (-2)*np.eye(p,p))
    
    Xi = U1.dot(X) - U0
        
    return Xi
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# Polar light retraction
# PL_U0 (Xi ) = (U0exp(U0'*Xi) + (I-U0U0')*Xi)(In + XiT(I-U0U0')Xi )^−1/2
# 
# Input arguments      
#          U0    : base point on St(n,p)
#          Xi : tangent vector in T_U0 St(n,p)
# Output arguments
#          U1    : PL_U0(Xi)
#------------------------------------------------------------------------------
def Stiefel_PL_ret(U0, Xi):
#------------------------------------------------------------------------------
    # get dimensions
    n,p = U0.shape
    
    A = np.dot(U0.T,Xi)  # horizontal component
    
    # eigenvalue decomposition
    # to-do: make this real Schur form
    Lambda, N = linalg.eig(A)
    
    D  = np.exp(Lambda) - Lambda
    NDNH = np.dot(N*D, N.conj().T)
    NDNH = NDNH.real # output must be real
    U1 = np.dot(U0, NDNH) + Xi
    
    
    # compute matrix square root
    # QR of (I-U0U0^T)Xi = Xi - U0*A
    S = np.linalg.qr(Xi-U0.dot(A), mode='r')
    S = np.eye(p) + np.dot(S.T, S)    
    
    S = scipy.linalg.sqrtm(scipy.linalg.inv(S))
        
    # assemble U1
    U1 = U1.dot(S)
    return U1
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Polar light inverse retraction
# PL_inv_U0 (U1) = (U0(logm(MR') -MR') + URS^-1R'
# where MSR' = U0'*U1 (SVD)
# Input arguments      
#          U0  : base point on St(n,p)
#          U1  : end point
# Output arguments
#          Xi  : tangent vector
#------------------------------------------------------------------------------
def Stiefel_PL_inv_ret(U0, U1):
    # get dimensions
    n,p = U0.shape
    
    M, S, RT = scipy.linalg.svd(np.dot(U0.T,U1),\
                               full_matrices=True,\
                               compute_uv=True,\
                               overwrite_a=True)
            
    MRT = M.dot(RT)
    
    RinvSRT = np.dot( (RT.T*(1./S)), RT)
    
    # assemble Xi
    Xi = U0.dot((linalg.logm(MRT) - MRT)) + U1.dot(RinvSRT)
    #[LogMRT , flag_negval] = StAux.SchurLog(MRT)
    #Xi = np.dot(U0, (LogMRT - MRT)) + np.dot(U1, RinvSRT)
    return Xi







#******************************************************************************
#  ||
#  ||     Down here: testing 
# \  /
#  \/
#******************************************************************************

do_tests = 1
if do_tests:
    
    # set dimensions
    n = 1000
    p = 200
    
    #for the Euclidean metric: alpha = -0.5
    #for the Canonical metric: alpha = 0.0
    metric_alpha = -0.0

    # set number of random experiments
    runs = 1
    dist = 0.5*np.pi

    #initialize
    time_array  = np.zeros((2,))
    is_equal    = np.zeros((2,))
    
    for j in range(runs):
        #----------------------------------------------------------------------
        #create random stiefel data
        U0, U1, Xi = Stiefel_Exp_Log.create_random_Stiefel_data(n, p, dist, metric_alpha)
        #----------------------------------------------------------------------
        t_start = time.time()         
        U1_pf   = Stiefel_PF_ret(U0, Xi)
        t_end   = time.time()        
        t_pf    = t_end-t_start
        
        Xi_pfi  = Stiefel_PF_inv_ret(U0, U1_pf)

        
        time_array[0] = time_array[0] + t_pf
        is_equal[0]   = is_equal[0]   + np.linalg.norm((Xi-Xi_pfi), 'fro')
        

        t_start = time.time()
        U1_pl   = Stiefel_PL_ret(U0, Xi)
        t_end   = time.time()
        t_pl    = t_end-t_start
        
        Xi_pli  = Stiefel_PL_inv_ret(U0, U1_pl)

        
        time_array[1] = time_array[1] + t_pl
        is_equal[1]   = is_equal[1]   + np.linalg.norm((Xi-Xi_pli), 'fro')

    
    print('time for polar factor retraction: ', time_array[0])
    print('normcheck', is_equal[0]/runs)
    print('time for polar light retraction: ', time_array[1])
    print('normcheck', is_equal[1]/runs)  
    
# End: if do_tests










