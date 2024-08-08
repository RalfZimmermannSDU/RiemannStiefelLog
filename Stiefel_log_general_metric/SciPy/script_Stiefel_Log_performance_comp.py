#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#
# Basic performance comparison for the various Stiefel log methods:
#
# Compute the average iteration count for the Stiefel Log for
# input data that are a preselected Riemannian distance apart.
#
#
# This script is an illustrative example for 
# solving the local geodesic endpoint problem
#    = computing the Riemannian logarithm on the Stiefel manifold.
#
# The algorithms work for a one-parameter family of metrics, including the 
# Euclidean and the canonical metric.
# The canonical metric allows for special algorithmic treatment.
# For all other metrics, a tailored shooting method is invoked.
#
# For theoretical background and description of the algorithms, see
#
# R. Zimmermann, K. H\"uper.
# "Computing the Riemannian logarithm on the Stiefel manifold: 
#  metrics, methods and performance", arXiv:2103.12046, March 2022
#
# If you make use of these methods, please cite the aforementioned reference.
#
#
# @author: Ralf Zimmermann, IMADA, SDU Odense
#------------------------------------------------------------------------------


import scipy
import numpy
import time

# local module
import Stiefel_Exp_Log as StEL



    
# set dimensions
n = 90
p = 30

#for the Euclidean metric: alpha = -0.5
#for the Canonical metric: alpha = 0.0
metric_alpha = 0.0

# set number of random experiments
runs = 10
dist = 0.8*numpy.pi
tau =  1.0e-11

print('***')
print('Executing', runs, 'random run(s) for computing the Stiefel log')
print('')
print('Parameter settings:')
print('Dimensions        (n,p) = (', n, ',', p,')')
print('Riemannian metric alpha = ', metric_alpha)
print('Riemannian dist(U0, U1) = ', dist, '= ', dist/numpy.pi, 'pi')
print('')
print('***')

# Methods to be compared
Alg        = 1
AlgCay     = 1
AlgCaySylv = 1
Shoot      = 1
Shoot4     = 1

#initialize
iters_array = numpy.zeros((5,))
time_array  = numpy.zeros((5,))
is_equal    = numpy.zeros((5,))
nonconv_counter = numpy.zeros((5,))

for j in range(runs):
    #----------------------------------------------------------------------
    #create random stiefel data
    U0, U1, Delta = StEL.create_random_Stiefel_data(n, p, dist, metric_alpha)
    #----------------------------------------------------------------------

    #----------------------------------------------------------------------
    # Method 0: algebraic Stiefel log
    if Alg and abs(metric_alpha)<1.0e-13:
        t_start = time.time()
        Delta_rec, conv_hist_Alg = StEL.Stiefel_Log_alg(U0, U1, tau,\
                                                        0,0,0)
        t_end = time.time()
        print('algebraic Stiefel log converged after',\
              len(conv_hist_Alg), ' iterations.')
            
        
        time_array[0] =  time_array[0] + t_end-t_start
        check_accuray = scipy.linalg.norm(Delta_rec-Delta, 1)
        is_equal[0]   = is_equal[0] + check_accuray
        iters_array[0]= iters_array[0] + len(conv_hist_Alg)
        if check_accuray >1.0e-9:
            nonconv_counter[0] = nonconv_counter[0] + 1
    else:
        conv_hist_Alg = [0]
    

    #----------------------------------------------------------------------
     
    #----------------------------------------------------------------------
    # Method 1: algebraic Stiefel log plus Cayley
    if AlgCay and abs(metric_alpha)<1.0e-13:
        t_start = time.time()
        Delta_rec, conv_hist_AlgCay = StEL.Stiefel_Log_alg(U0, U1, tau,\
                                                           0,1,0)
        t_end = time.time()
        
        print('alg. Stiefel log+Cay  converged after',\
              len(conv_hist_AlgCay), ' iterations.')
        
        time_array[1] =  time_array[1] + t_end-t_start
        check_accuray = scipy.linalg.norm(Delta_rec-Delta, 1)
        is_equal[1]   = is_equal[1] + check_accuray
        iters_array[1]= iters_array[1] + len(conv_hist_AlgCay)
        
        if check_accuray >1.0e-9:
            nonconv_counter[1] = nonconv_counter[1] + 1
    else:
        conv_hist_Alg = [0]
    

    #----------------------------------------------------------------------

    #----------------------------------------------------------------------
    # Method 2: algebraic Stiefel log plus Cayley plus Sylv
    if AlgCaySylv and abs(metric_alpha)<1.0e-13:
        t_start = time.time()
        Delta_rec, conv_hist_AlgCaySylv = StEL.Stiefel_Log_alg(U0, U1, tau,\
                                                               0,1,1)
        t_end = time.time()
        
        print('alg. Stiefel log+Cay+Sylv conv. after',\
              len(conv_hist_AlgCaySylv), ' iterations.')
        
        time_array[2] =  time_array[2] + t_end-t_start
        
        check_accuray = scipy.linalg.norm(Delta_rec-Delta, 1)
        is_equal[2]   = is_equal[2] + check_accuray
        iters_array[2]= iters_array[2] + len(conv_hist_AlgCaySylv)
        
        if check_accuray >1.0e-9:
            nonconv_counter[2] = nonconv_counter[2] + 1
        
    else:
        conv_hist_Alg = [0]


    #----------------------------------------------------------------------

    #----------------------------------------------------------------------
    # Method 3: shooting method on two steps
    if Shoot:
        unit_int =  numpy.linspace(0.0,1.0,2)
        t_start = time.time()
        Delta_rec, conv_hist_pS = StEL.Stiefel_Log_p_Shooting_uni(U0,\
                                                                  U1,\
                                                                  unit_int,\
                                                                  tau,\
                                                                  metric_alpha)
        t_end = time.time()
        
        print('p-Shooting unified method on', len(unit_int), 'steps converged in ',\
              len(conv_hist_pS), ' iterations')
        
        time_array[3] =  time_array[3] + t_end-t_start
        check_accuray = scipy.linalg.norm(Delta_rec-Delta, 1)
        is_equal[3]   = is_equal[3] + check_accuray
        iters_array[3]= iters_array[3] + len(conv_hist_pS)
        if check_accuray >1.0e-9:
            nonconv_counter[3] = nonconv_counter[3] + 1
        
    else:
        conv_hist_pS = [0]

    #----------------------------------------------------------------------
    
    #----------------------------------------------------------------------
    # Method 4: shooting method on four steps
    if Shoot4:  
        unit_int = numpy.linspace(0.0, 1.0, 4)
        t_start = time.time()
        Delta_rec, conv_hist_pS4 = StEL.Stiefel_Log_p_Shooting_uni(U0,\
                                                                   U1,\
                                                                   unit_int,\
                                                                   tau,\
                                                                   metric_alpha)
        t_end = time.time()
        
        print('p-Shooting unified method on', len(unit_int), 'steps converged in ',\
              len(conv_hist_pS4), ' iterations')
        
        time_array[4] =  time_array[4] + t_end-t_start
        check_accuray = scipy.linalg.norm(Delta_rec-Delta, 1)
        is_equal[4]   = is_equal[4] + check_accuray
        iters_array[4]= iters_array[4] + len(conv_hist_pS4)
        if check_accuray >1.0e-9:
            nonconv_counter[4] = nonconv_counter[4] + 1
    else:
        conv_hist_pS4 = [0]
    #----------------------------------------------------------------------
    # End loop over runs

# average time and iteration count
print('')
print('The average iteration count of the various methods is:')
iters_array = iters_array/runs
print(iters_array)

print('')
print('The average cpu time of the various methods is:')
time_array = time_array/runs
print(time_array)

print('')
print('The average reconstruction accuracy of the various methods is:')
is_equal = is_equal/runs
print(is_equal)
print('')
print('Were there any runs that did not produce the correct result?')
print(nonconv_counter)
# End
