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
n = 600
p = 250

#for the Euclidean metric: alpha = -0.5
#for the Canonical metric: alpha = 0.0
metric_alpha = -0.0

# set number of random experiments
runs = 10
dist = 0.8*scipy.pi
tau =  1.0e-11

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
        time_array[0] =  time_array[0] + t_end-t_start
        is_equal[0]   = is_equal[0] + scipy.linalg.norm(Delta_rec-Delta, 1)
    else:
        conv_hist_Alg = [0]
    
    iters_array[0] = iters_array[0] + len(conv_hist_Alg)
    #----------------------------------------------------------------------
     
    #----------------------------------------------------------------------
    # Method 1: algebraic Stiefel log plus Cayley
    if AlgCay and abs(metric_alpha)<1.0e-13:
        t_start = time.time()
        Delta_rec, conv_hist_AlgCay = StEL.Stiefel_Log_alg(U0, U1, tau,\
                                                           0,1,0)
        t_end = time.time()
        time_array[1] =  time_array[1] + t_end-t_start
        is_equal[1]   = is_equal[1] + scipy.linalg.norm(Delta_rec-Delta, 1)
    else:
        conv_hist_Alg = [0]
    
    iters_array[1] = iters_array[1] + len(conv_hist_AlgCay)
    #----------------------------------------------------------------------

    #----------------------------------------------------------------------
    # Method 2: algebraic Stiefel log plus Cayley plus Sylv
    if AlgCaySylv and abs(metric_alpha)<1.0e-13:
        t_start = time.time()
        Delta_rec, conv_hist_AlgCaySylv = StEL.Stiefel_Log_alg(U0, U1, tau,\
                                                               0,1,1)
        t_end = time.time()
        time_array[2] =  time_array[2] + t_end-t_start
        is_equal[2]   = is_equal[2] + scipy.linalg.norm(Delta_rec-Delta, 1)
    else:
        conv_hist_Alg = [0]

    iters_array[2] = iters_array[2] + len(conv_hist_AlgCaySylv)
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
        time_array[3] =  time_array[3] + t_end-t_start
        is_equal[3]   = is_equal[3] + scipy.linalg.norm(Delta_rec-Delta, 1)
    else:
        conv_hist_pS = [0]

    iters_array[3] = iters_array[3] + len(conv_hist_pS)
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
        time_array[4] =  time_array[4] + t_end-t_start
        is_equal[4] = is_equal[4] + scipy.linalg.norm(Delta_rec-Delta, 1)
    else:
        conv_hist_pS4 = [0]
    
    iters_array[4] = iters_array[4] + len(conv_hist_pS4)
    #----------------------------------------------------------------------
    # End loop over runs

# average time and iteration count

print('The average iteration count of the various methods is:')
iters_array = iters_array/runs
print(iters_array)
    
print('The average cpu time of the various methods is:')
time_array = time_array/runs
print(time_array)

print('The average reconstruction accuracy of the various methods is:')
is_equal = is_equal/runs
print(is_equal)
    
# End: if do_tests
