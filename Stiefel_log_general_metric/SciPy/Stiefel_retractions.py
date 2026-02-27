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

import sys

import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Polar factor retraction
# PF_U0 (Xi ) = (U0 + Xi)(In + XiT Xi )^-1/2
# 
# Input arguments      
#          U0    : base point on St(n,p)
#          Xi : tangent vector in T_U0 St(n,p)
# Output arguments
#          U1    : PF_U0(Xi)
#------------------------------------------------------------------------------
def Stiefel_PF_ret(U0, Xi):
#------------------------------------------------------------------------------
    # get dimensions
    n,p = U0.shape
    
    # QR decomposition of Xi,
    # only R is needed
    S = np.linalg.qr(Xi, mode='r')
    
    fast = 1
    if fast:
        # symmetric EVD 
        STS = np.dot(S.T, S)  
        Lambda, V  = scipy.linalg.eigh(STS)
        # sqrt( 1/(1+lambda)), addition is elementwise
        Lambda = np.sqrt(1/(Lambda + 1))
        STS = np.dot(V*Lambda, V.T)
    else:
        # SVD, slower, but more robust
        M, Sing, VT = scipy.linalg.svd(S,\
                               full_matrices=True,\
                               compute_uv=True,\
                               overwrite_a=True)
        # sqrt( 1/(1+sing^2)), addition is elementwise
        Sing = np.sqrt(1/(Sing*Sing + 1))
        STS  = np.dot(VT.T*Sing, VT)
    
    
    
    #S = np.eye(p,p) + np.dot(R.transpose(), R) #np.dot(Xi.transpose(), Xi);
    # compute matrix square root
    #S = linalg.sqrtm(scipy.linalg.inv(S))
        
    # perform U1 = U0*M + Q*N
    U1 = np.dot((U0+Xi),STS)
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
# PL_U0 (Xi ) = (U0exp(U0'*Xi) + (I-U0U0')*Xi)(I + XiT(I-U0U0')Xi )^−1/2
# 
# Input arguments      
#          U0    : base point on St(n,p)
#          Xi : tangent vector in T_U0 St(n,p)
# Output arguments
#          U1    : PL_U0(Xi)
#------------------------------------------------------------------------------
def Stiefel_PL_ret(U0, Xi, mode=1):
#------------------------------------------------------------------------------
    # get dimensions
    n,p = U0.shape
    
    A = np.dot(U0.T,Xi)  # horizontal component
    
    
    if mode == 1:
        # general purpose matrix exponential
        # seems to be faster than homebrew Schur exp
        expA_A  = scipy.linalg.expm(A) - A
    elif mode == 2:
        print('PL mode2')
        # Cayley trafo
        expA_A  = StAux.Cayley(A) - A
    else:
        # eigenvalue decomposition
        # to-do: make this real Schur form
        Lambda, N = linalg.eig(A)

        D  = np.exp(Lambda) - Lambda
        expA_A = np.dot(N*D, N.conj().T)
        expA_A = expA_A.real # output must be real
   
        
        
    # assemble A
    U1      = np.dot(U0, expA_A) + Xi
    
    # compute matrix square root
    # QR of (I-U0U0^T)Xi = Xi - U0*A
    S    = np.linalg.qr(Xi-U0.dot(A), mode='r')   
    
    fast = 1
    if fast:
        # symmetric EVD 
        Lambda, V  = scipy.linalg.eigh(np.dot(S.T, S))
        # sqrt( 1/(1+lambda)), addition is elementwise
        Lambda = np.sqrt(1/(Lambda + 1))
        STS = np.dot(V*Lambda, V.T)
    else:
        # SVD, slower, but more robust
        M, Sing, VT = scipy.linalg.svd(S,\
                               full_matrices=True,\
                               compute_uv=True,\
                               overwrite_a=True)
        # sqrt( 1/(1+sing^2)), addition is elementwise
        Sing = np.sqrt(1/(Sing*Sing + 1))
        STS  = np.dot(VT.T*Sing, VT)
   
    # assemble U1
    U1 = U1.dot(STS)
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
def Stiefel_PL_inv_ret(U0, U1, mode=1):
    # get dimensions
    n,p = U0.shape
    
    M, S, RT = scipy.linalg.svd(np.dot(U0.T,U1),\
                               full_matrices=True,\
                               compute_uv=True,\
                               overwrite_a=True)
            
    MRT = M.dot(RT)
    
    RinvSRT = np.dot( (RT.T*(1./S)), RT)
    
    # assemble Xi
    if mode == 1:
        Xi = U0.dot((linalg.logm(MRT) - MRT)) + U1.dot(RinvSRT)
    else:
        print('PL inv mode2')
        # Cayley trafo for replacing the logm
        Xi = U0.dot((StAux.Cayley_inv(MRT) - MRT)) + U1.dot(RinvSRT)
    return Xi







#******************************************************************************
#  ||
#  ||     Down here: testing 
# \  /
#  \/
#******************************************************************************

do_tests = 0
if do_tests:
    
    # set dimensions
    n = 4000
    p = 1000
    
    #for the Euclidean metric: alpha = -0.5
    #for the Canonical metric: alpha =  0.0
    metric_alpha = -0.5

    # set number of random experiments
    runs = 10
    dist = 0.5*np.pi

    #initialize
    time_array  = np.zeros((2,))
    is_equal    = np.zeros((2,))
    
    #----------------------------------------------------------------------
    #create random stiefel data
    U0, U1, Xi = Stiefel_Exp_Log.create_random_Stiefel_data(n, p, dist, metric_alpha)
    #----------------------------------------------------------------------
  
    for j in range(runs):
    
        if 0: # test cayley trafo
            A = np.dot(U0.T, Xi)
            A = 0.5*(A-A.T)
        
            # cayley test
            t_start    = time.time() 
            Q = StAux.Cayley(A)
            Ac= StAux.Cayley_inv(Q)
            t_end      = time.time()        
            t_cay      = t_end-t_start
            print('t cay:', t_cay, 's', 'norm cayley check', linalg.norm(A-Ac))
      
        
        # check if PF_inv(PF(Xi)) = Xi
        t_start = time.time()      
        U1_pf   = Stiefel_PF_ret(U0, Xi)
        Xi_pfi  = Stiefel_PF_inv_ret(U0, U1_pf)
        t_end   = time.time()        
        t_pf    = t_end-t_start
        
        time_array[0] = time_array[0] + t_pf
        is_equal[0]   = is_equal[0]   + np.linalg.norm((Xi-Xi_pfi), 'fro')
        

        # check if PL_inv(PL(Xi)) = Xi
        t_start = time.time()
        U1_pl   = Stiefel_PL_ret(U0, Xi)      
        Xi_pli  = Stiefel_PL_inv_ret(U0, U1_pl)
        t_end   = time.time()
        t_pl    = t_end-t_start
               
        time_array[1] = time_array[1] + t_pl
        is_equal[1]   = is_equal[1]   + np.linalg.norm((Xi-Xi_pli), 'fro')
        #sys.exit()
    
    print('time for polar factor retraction: ', time_array[0]/runs)
    print('normcheck', is_equal[0]/runs)
    print('time for polar light retraction: ', time_array[1]/runs)
    print('normcheck', is_equal[1]/runs)  
    
    # compare to geodesic
    # we still have the data triple
    # U0, U1, Xi 
    # with 
    # exp_U0(Xi) = U1.
    #
    # geodesic:  exp_U0)(t*Xi), t \in [0,1]
    # 
    # any retraction R_U0 connecting the same endpoints:
    # (1) compute R_U0^{-1}(U1). This yields tangent vector Xi_R.
    # with 
    #       R_U0(Xi_R) = U1.
    # (2) evaluate retraction curve
    #       R_U0(t*Xi), t\in [0,1]
    # (3) compute error between geodesic and retraction curve

    # tangent for polar factor retraction
    Xi_pfi  = Stiefel_PF_inv_ret(U0, U1)
    
    # tangent for polar light retraction
    Xi_pli  = Stiefel_PL_inv_ret(U0, U1)
    
    # discrete unit interval
    num_t = 4
    I_unit = np.linspace(0.0, 1.0, num=num_t)
    
    
    errors_geo_approx = np.zeros((num_t,2))
    
    for k in range(num_t):
        tk      = I_unit[k]
        # geodesic at tk
        Exp_tk  = Stiefel_Exp_Log.Stiefel_Exp(U0, tk*Xi, metric_alpha)
        # PF retraction at tk
        PF_tk   = Stiefel_PF_ret(U0, tk*Xi_pfi)
        # PL retraction at tk
        PL_tk   = Stiefel_PL_ret(U0, tk*Xi_pli)

        error_PF= np.linalg.norm((Exp_tk-PF_tk), 'fro')
        error_PL= np.linalg.norm((Exp_tk-PL_tk), 'fro')
        errors_geo_approx[k,0] = error_PF
        errors_geo_approx[k,1] = error_PL
    
    print('Max errors are:')
    print(np.max(errors_geo_approx[:,0]), ' (polar factor)')
    print(np.max(errors_geo_approx[:,1]), ' (polar light)')
    do_plot = True
    if do_plot:
        plt.rcParams.update({'font.size': 40})

        line_err_PF, = plt.plot(I_unit, errors_geo_approx[:,0], 'r-', linewidth=3, label = 'errors PF')
        line_err_PL, = plt.plot(I_unit, errors_geo_approx[:,1], 'k-.', linewidth=3, label = 'errors PL')
    
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('Errors')
    plt.show()
        
        
# End: if do_tests










