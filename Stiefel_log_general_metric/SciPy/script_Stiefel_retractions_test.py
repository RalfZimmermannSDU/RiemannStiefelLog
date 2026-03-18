#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:13:20 2026

@author: zimmermann
"""

import numpy as np
from   scipy import linalg
import Stiefel_Exp_Log     as StEL
import Stiefel_Aux         as StAux
import Stiefel_retractions as StRet
import time

import matplotlib.pyplot as plt



#******************************************************************************
#  ||    Experiment associated with 
#  ||    Table 1, Fig. 3, Fig. 4 (to-do update references) of
#  ||    "AN NEW POLAR FACTOR RETRACTION ON THE STIEFEL
# \  /    MANIFOLD WITH CLOSED-FORM INVERSE"
#  \/     
#******************************************************************************


    
# set dimensions
n = 1000
p = 400
    
#for the Euclidean metric: alpha = -0.5
#for the Canonical metric: alpha =  0.0
metric_alpha = -0.5

# set number of random experiments
runs = 100
dist = 0.5*np.pi

#initialize
time_array  = np.zeros((2,))
is_equal    = np.zeros((2,))

#----------------------------------------------------------------------
#create random stiefel data
U0, U1, Xi = StEL.create_random_Stiefel_data(n, p, dist, metric_alpha)
#----------------------------------------------------------------------

for j in range(runs):
    print('run ', j, ' of ', runs)
    if 0: # test cayley trafo
        A = np.dot(U0.T, Xi)
        A = 0.5*(A-A.T)
        
        # cayley test
        t_start    = time.time() 
        Q = StAux.Cayley(A)
        Ac= StAux.Cayley_inv(Q)
        Qc= StAux.Cayley(Ac)
        t_end      = time.time()        
        t_cay      = t_end-t_start
        print('t cay:', t_cay, 's', 'norm cayley check', linalg.norm(A-Ac))
        print('norm cayley check 2', linalg.norm(Q-Qc))
    
    # check if PF_inv(PF(Xi)) = Xi     
    U1_pf   = StRet.Stiefel_PF_ret(U0, Xi)
    
    t_start = time.time()    # measure time of inverse retraction
    Xi_pfi  = StRet.Stiefel_PF_inv_ret(U0, U1_pf)
    t_end   = time.time()        
    t_pf    = t_end-t_start
    
    time_array[0] = time_array[0] + t_pf
    is_equal[0]   = is_equal[0]   + np.linalg.norm((Xi-Xi_pfi), 'fro')
    

    # check if PL_inv(PL(Xi)) = Xi
    mode = 1; # 1: expm, logm, 2: Cay, Cay_inv
    U1_pl   = StRet.Stiefel_PL_ret(U0, Xi, mode)  
    
    t_start = time.time()    # measure time of inverse retraction
    Xi_pli  = StRet.Stiefel_PL_inv_ret(U0, U1_pl, mode)
    t_end   = time.time()
    t_pl    = t_end-t_start
    
    time_array[1] = time_array[1] + t_pl
    is_equal[1]   = is_equal[1]   + np.linalg.norm((Xi-Xi_pli), 'fro')
    #sys.exit()
    
print('time for polar factor retraction: ', time_array[0]/runs)
print('normcheck', is_equal[0]/runs)
print('time for polar light retraction: ', time_array[1]/runs)
print('normcheck', is_equal[1]/runs)  
    
    
#**************************************************************************
#
# ALL ACTIONS ARE PERFORMED UNDER THE EUCLIDEAN METRIC
#
# Experiment 1: compare retractions to geodesic
# we still have the data triple
# U0, U1, Xi 
# with 
# exp_U0(Xi) = U1.
#
# geodesic:  Exp_U0(t*Xi), t \in [0,1]
# 
# any retraction R_U0 connecting the same endpoints:
# (1) compute R_U0^{-1}(U1). This yields tangent vector Xi_R.
# with 
#       R_U0(Xi_R) = U1.
# (2) evaluate retraction curve
#       R_U0(t*Xi), t\in [0,1]
# (3) compute error between geodesic and retraction curve
#**************************************************************************
    
    
for mode in range(1,3):  # this gives mode in {1,2}
    # mode=1: matrix exp/log, mode 2: Cay, Cay^{-1}
    
    # tangent for polar factor retraction
    Xi_pfi  = StRet.Stiefel_PF_inv_ret(U0, U1)
    
    # tangent for polar light retraction
    Xi_pli  = StRet.Stiefel_PL_inv_ret(U0, U1, mode)
    
    # discrete unit interval
    num_t = 51
    I_unit = np.linspace(0.0, 1.0, num=num_t)
    
    
    errors_geo_approx = np.zeros((num_t,2))
    
    # for later use, store the geodesic curve points
    geo_t = np.zeros((num_t, n,p))
    
    for k in range(num_t):
        tk      = I_unit[k]
        # geodesic at tk
        geo_t[k,:,:]  = StEL.Stiefel_Exp(U0, tk*Xi, metric_alpha)
        # PF retraction at tk
        PF_tk   = StRet.Stiefel_PF_ret(U0, tk*Xi_pfi)
        # PL retraction at tk
        PL_tk   = StRet.Stiefel_PL_ret(U0, tk*Xi_pli, mode)

        error_PF= np.linalg.norm((geo_t[k,:,:]-PF_tk), 'fro')
        error_PL= np.linalg.norm((geo_t[k,:,:]-PL_tk), 'fro')
        errors_geo_approx[k,0] = error_PF
        errors_geo_approx[k,1] = error_PL
    
    print('Max errors mode ', mode, ' are:')
    print(np.max(errors_geo_approx[:,0]), ' (polar factor)')
    print(np.max(errors_geo_approx[:,1]), ' (polar light)')
    do_plot = True
    if do_plot:
        plt.rcParams.update({'font.size': 40})

        line_err_PF, = plt.plot(I_unit, errors_geo_approx[:,0], 'r-', linewidth=3, label = 'errors PF')
        line_err_PL, = plt.plot(I_unit, errors_geo_approx[:,1], 'b-.', linewidth=3, label = 'errors PL')
    
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('Errors')
    plt.show()
        
    #**************************************************************************
    #
    # ALL ACTIONS ARE PERFORMED UNDER THE EUCLIDEAN METRIC
    #
    # Experiment 2: compare inverse retractions to Riemannian logarithm
    # we still have the data triple
    # U0, U1, Xi 
    # with 
    #  Exp_U0(Xi) = U1.
    #
    # Form the previous experiment, we have the Stiefel geodesic
    #  geo_t: t \to Exp_U0(t*Xi)
    # 
    # Here, we take this curve and pull it back to the coordinate domain
    # by using an inverse retraction R_U0^{-1}.
    # When the logarithm is used, this should give the straight line.
    #
    # ALL ACTIONS ARE PERFORMED UNDER THE EUCLIDEAN METRIC
    #**************************************************************************
    errors_coord_approx = np.zeros((num_t,3))
    # 1st column: errors under Riemann log (should be close to 0)
    # 2nd column: errors under inverse polar factor retraction
    # 3rd column: errors under inverse polar light retraction
   
    # for later use, store the Riemann log coordinate curve points
    Log_coord_curve_t = np.zeros((num_t, n,p))
    
    # measure time for each loop
    t_start = time.time()
    for k in range(num_t):
        tk      = I_unit[k]
        # geodesic at tk
        tau = 1.0e-11 # convergence threshold for Riemann log
        Log_coord_curve_t[k,:,:], conv_log  = StEL.Stiefel_Log(U0, geo_t[k,:,:], tau, metric_alpha)
        errors_coord_approx[k,0] = np.linalg.norm((Log_coord_curve_t[k,:,:]-tk*Xi), 'fro')
    t_end   = time.time()
    t_rl    = t_end-t_start    # time under Riemann log
    
    t_start = time.time()
    for k in range(num_t):
        tk      = I_unit[k]
        # PF retraction at tk
        PF_coord_curve_tk       = StRet.Stiefel_PF_inv_ret(U0, geo_t[k,:,:])
        errors_coord_approx[k,1]= np.linalg.norm((Log_coord_curve_t[k,:,:]-PF_coord_curve_tk), 'fro')
    t_end   = time.time()
    t_pf    = t_end-t_start    # time under Riemann log 

    t_start = time.time()  
    for k in range(num_t):
        tk      = I_unit[k]
        # PL retraction at tk
        PL_coord_curve_tk       = StRet.Stiefel_PL_inv_ret(U0, geo_t[k,:,:], mode)
        errors_coord_approx[k,2]= np.linalg.norm((Log_coord_curve_t[k,:,:]-PL_coord_curve_tk), 'fro')
    t_end   = time.time()
    t_pl    = t_end-t_start    # time under Riemann log

    
    print('time for Riemann coord curve     :', t_rl)
    print('time for polar factor coord curve:', t_pf)
    print('time for polar light        curve:', t_pl)       
    print('Max errors are:')
    print(np.max(errors_coord_approx[:,0]), ' (Riemann baseline)')    
    print(np.max(errors_coord_approx[:,1]), ' (polar factor)')
    print(np.max(errors_coord_approx[:,2]), ' (polar light)')
    do_plot = True
    if do_plot:
        plt.rcParams.update({'font.size': 40})
        line_err_RL, = plt.plot(I_unit, errors_coord_approx[:,0], 'k--', linewidth=3, label = 'errors RL')
        line_err_PF, = plt.plot(I_unit, errors_coord_approx[:,1], 'r-',  linewidth=3, label = 'errors PF')
        line_err_PL, = plt.plot(I_unit, errors_coord_approx[:,2], 'b-.', linewidth=3, label = 'errors PL')
    
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('Errors')
    plt.show()

# End: if do_tests