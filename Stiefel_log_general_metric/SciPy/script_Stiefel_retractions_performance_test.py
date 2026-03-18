#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:13:20 2026

@author: zimmermann
"""

import numpy as np
import Stiefel_Exp_Log     as StEL
import Stiefel_retractions as StRet
import time



#******************************************************************************
#  ||    Experiment associated with Figure 1 (to-do update reference) of
#  ||    "AN NEW POLAR FACTOR RETRACTION ON THE STIEFEL
# \  /    MANIFOLD WITH CLOSED-FORM INVERSE"
#  \/     (The figure is produced with matlab based on the time_array
#          that is computed in this script.)
#******************************************************************************


    
# set dimensions
n = 10000

    
#for the Euclidean metric: alpha = -0.5
#for the Canonical metric: alpha =  0.0
metric_alpha = -0.5
    
# set number of random experiments
runs = 10
dist = 1.0*np.pi

#initialize
time_array  = np.zeros((6,))
    # time_array[0,:] : timings full PF 
    # time_array[1,:] : timings full PF inverse
    # time_array[2,:] : timings polar light 
    # time_array[3,:] : timings polar light inverse 
    # time_array[4,:] : timings polar light Cayley
    # time_array[5,:] : timings polar light Cayley inverse 


for p in [500,1000,1500,2000]:
    for j in range(runs):
        print('Dim p=', p, 'Run ', j)
        #----------------------------------------------------------------------
        #create random stiefel data
        U0, U1, Xi = StEL.create_random_Stiefel_data(n, p, dist, metric_alpha)
        #----------------------------------------------------------------------
        
        # classical polar factor retraction
        t_start = time.time()      
        U1_pf   = StRet.Stiefel_PF_ret(U0, Xi)
        t_end   = time.time() 
        time_array[0] = time_array[0] + (t_end-t_start)
        
        t_start = time.time()    
        Xi_pfi  = StRet.Stiefel_PF_inv_ret(U0, U1_pf)
        t_end   = time.time() 
        time_array[1] = time_array[1] + (t_end-t_start) 
        
        # polar light
        mode = 1 # i.e. "use expm, logm
        t_start = time.time()      
        U1_pl   = StRet.Stiefel_PL_ret(U0, Xi,mode)
        t_end   = time.time() 
        time_array[2] = time_array[2] + (t_end-t_start)
        
        t_start = time.time()    
        Xi_pli  = StRet.Stiefel_PL_inv_ret(U0, U1_pf,mode)
        t_end   = time.time() 
        time_array[3] = time_array[3] + (t_end-t_start) 
        
        # polar light plus Cayley
        mode = 2 # i.e. "use Cay, Cay_inv
        t_start = time.time()      
        U1_pl   = StRet.Stiefel_PL_ret(U0, Xi,mode)
        t_end   = time.time() 
        time_array[4] = time_array[4] + (t_end-t_start)
        
        t_start = time.time()    
        Xi_pli  = StRet.Stiefel_PL_inv_ret(U0, U1_pf,mode)
        t_end   = time.time() 
        time_array[5] = time_array[5] + (t_end-t_start)   
        #sys.exit()
        
    print('Dim:',p, ':time for polar factor     retraction    : ', time_array[0]/runs)
    print('Dim:',p, ':time for polar factor inv retraction    : ', time_array[1]/runs)
    print('Dim:',p, ':time for polar light      retraction    : ', time_array[2]/runs)
    print('Dim:',p, ':time for polar light  inv retraction    : ', time_array[3]/runs)
    print('Dim:',p, ':time for polar light      retraction Cay: ', time_array[4]/runs)
    print('Dim:',p, ':time for polar light  inv retraction Cay: ', time_array[5]/runs)
