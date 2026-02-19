# *****************************************************************************
#
# This file implements the numerical experiment
# that features in Section 5.4 of
# 
# HERMITE INTERPOLATION AND DATA PROCESSING ERRORS ON RIEMANNIAN MATRIX MANIFOLDS
# SIAM J. SCI . COMPUT. Vol. 42, No. 5, pp. A2593--A2619
#
# @author: R. Zimmermann
#
# *****************************************************************************

import numpy as np
import math

import matplotlib.pyplot as plt

import time
import sys

sys.path.append('../../../Stiefel_log_general_metric/SciPy/')
sys.path.append('../Stiefel_interp/')
sys.path.append( '../General_interp_tools/')

import snapshot_analytic_mat    as snam
import Stiefel_interp_funcs     as sifs
#import Stiefel_interp_funcs_retra     as sifs

import RBF_interp as RBF
import Hermite_interp as HI

# set dimension
n = 1001
p = 6
# choose RBF function
#RBF_fun = RBF.Invmultiquad_rbf
#RBF_fun = RBF.Multiquad_rbf
#RBF_fun = RBF.TPS_rbf
#RBF_fun = RBF.Cubic_rbf
RBF_fun = RBF.Invquad_rbf
#RBF_fun = RBF.Gauss_rbf

print( "")
print( "***********************************")
print( "Which experiments are to be performed?")
in_geo      = input('      Geodesic interpolation 0/1?: ')
in_tang_int = input(' Tangent space interpolation 0/1?: ')
in_Hermite  = input(' Cubic Hermite interpolation 0/1?: ')

#covert input string from keyboard to integers
do_geo = int(in_geo)
do_tang_int = int(in_tang_int)
do_Hermite  = int(in_Hermite)


# QR or SVD
key = 'SVD'

# which metric?
# use alpha =  0.0 for canonical 
# use alpha = -0.5 for Euclidean
alpha = 0.0

# create snapshots
# t-samples:
t_range = np.linspace(1.0, 4.0, p)


mu_samples = HI.ChebyRoots(1.7, 2.3, 6)
print("samples go here", mu_samples)

mu_range = np.linspace(mu_samples[0], mu_samples[-1], 100)

# create snapshot matrices of size n x p
U_matrix_list = np.zeros((len(mu_samples), n,p))
dU_matrix_list = np.zeros((len(mu_samples), n,p))
Sigma_list  = np.zeros((len(mu_samples), p))
dSigma_list = np.zeros((len(mu_samples), p))
V_matrix_list  = np.zeros((len(mu_samples), p,p))
dV_matrix_list = np.zeros((len(mu_samples), p,p))

# first, reference SVD for normalization purposes
# provide empty matrix as reference
RefU, RefS, RefV = snam.snapshot_analytic_mat2(t_range, mu_samples[int(len(mu_samples)/2)],\
                                               n, np.array([[None]]), 0, key)

for j in range(len(mu_samples)):
    U_matrix_list[j,:,:], dU_matrix_list[j,:,:], \
    Sigma_list[j,:], dSigma_list[j,:], \
    V_matrix_list[j,:,:], dV_matrix_list[j,:,:]  = \
                        snam.snapshot_analytic_mat2(t_range, mu_samples[j], n, RefU, 1, key)
        

# *****************************************************************************
# at this stage, each matrix
# Uj = U_matrix_list(j,:,:)
# is a  column-orthogonal matrix that
# depends smoothly on mu
# 
# next: compute list of reference values
# *****************************************************************************
U_true_list = np.zeros((len(mu_range), n, p))
S_true_list = np.zeros((len(mu_range), p))
V_true_list = np.zeros((len(mu_range), p, p))

for k in range(len(mu_range)):
    #trial point
    mu_star = mu_range[k]   
    # for comparison: true matrix
    # use same reference to align the coordinates
    U, S, V = snam.snapshot_analytic_mat2(t_range, mu_star, n, RefU, 0, key)
    U_true_list[k,:,:] = U
    S_true_list[k,:]   = S
    V_true_list[k,:,:] = V

# VISUAL CHECK, IF SVD IS ANALYTIC in MU
check_analytic = 0
if check_analytic:
    plt.rcParams.update({'font.size': 20})
    line_sigma_pm2,  = plt.plot(mu_range, S_true_list[:,p-3], 'k--', label = 'sigma p-2')
    line_sigma_pen,  = plt.plot(mu_range, S_true_list[:,p-2], 'b--', label = 'sigma p-1')
    line_sigma_min,  = plt.plot(mu_range, S_true_list[:,p-1], 'k--', linewidth=3, label = 'sigma p')
    
    plt.legend()
    plt.xlabel('mu')
    plt.ylabel('sigma')
    plt.show()




# *****************************************************************************
# perform interpolation:
# *****************************************************************************


# *****************************************************************************
# FIRST EXPERIMENT
# linear interpolation for "all" mu in sample range
# works only in 1D 
# *****************************************************************************
if do_geo:
    subspace_errors = np.zeros((len(mu_range),))
    ref_coeffs = list()
    rbf_coeffs = list()
    # PREPROCESSING 
    Deltas = sifs.Stiefel_geodesic_interp_pre(U_matrix_list, mu_samples, alpha)

    print('  ***   ')
    print('Start of geodesic interpolation')
    print('  ***   ')
    t_start = time.time()
    for k in range(len(mu_range)):
        #trial point
        mu_star = mu_range[k]
        #print("geo int at", mu_star)
        U_star = sifs.Stiefel_geodesic_interp(U_matrix_list,\
                                              Deltas,\
                                              mu_samples,\
                                              mu_star,\
                                              alpha)
        subspace_errors[k] = np.linalg.norm(U_true_list[k,:,:]-U_star, 'fro')/np.linalg.norm(U_true_list[k,:,:], 'fro')
    
    t_end = time.time()
    print('  ***   ')
    print('Geodesic interpolation finished in ', t_end-t_start, 's')
    print('  ***   ')
    
    

# *****************************************************************************
# THIRD EXPERIMENT
# RBF interpolation in tangent space
# * map all data to single tangent space
# * perform RBF interpolation in tangent space
# 
# *****************************************************************************
if do_tang_int:
    sample_sites = np.zeros((1, len(mu_samples)))
    sample_sites[0,:] = mu_samples
    subspace_errors_RBF = np.zeros((len(mu_range),))

    # PREPROCESSING: map all sample locations to tangent space
    Deltas = sifs.Stiefel_RBF_tang_interp_pre(U_matrix_list,\
                                              sample_sites,\
                                              alpha)

    # inverse of correlation matrix stays fixed
    Rinv = RBF.RBF_invmat(sample_sites, RBF_fun)
    print('  ***   ')
    print('Start of tangent space interpolation')
    print('  ***   ')
    t_start = time.time()
    # full interpolation
    for k in range(len(mu_range)):
        #trial point
        mu_star = mu_range[k]
        #print("tang int at", mu_star)
        U_star = sifs.Stiefel_RBF_tang_interp(U_matrix_list,\
                                              sample_sites,\
                                              Deltas,\
                                              mu_star,\
                                              Rinv,\
                                              RBF_fun,
                                              alpha)
    
        subspace_errors_RBF[k] = np.linalg.norm(U_true_list[k,:,:]-U_star, 'fro')/np.linalg.norm(U_true_list[k,:,:], 'fro')
    t_end = time.time()
    print('  ***   ')
    print('Tangent space interpolation finished in ', t_end-t_start, 's')
    print('  ***   ')
      


# *****************************************************************************
# Fifth EXPERIMENT
# Quasi-Cubic Hermite Interpolation
# 
# *****************************************************************************
if do_Hermite:
    mode = 1  #q-centered calculation
    subspace_errors_Hermite_q = np.zeros((len(mu_range),))

    # preprocessing of sample data
    Deltas, Vs_shifted = sifs.Stiefel_Hermite_interp_pre(U_matrix_list,\
                                                         dU_matrix_list,\
                                                         mu_samples,\
                                                         alpha,\
                                                         mode)
    print('  ***   ')
    print('Start of Hermite interpolation -q')
    print('  ***   ')
    t_start = time.time()
    # full interpolation
    for k in range(len(mu_range)):
        #trial point
        mu_star = mu_range[k]
        #print("Herm int at", mu_star)
        U_star = sifs.Stiefel_Hermite_interp(U_matrix_list,\
                                             dU_matrix_list,\
                                             Deltas,\
                                             Vs_shifted,\
                                             mu_samples,\
                                             mu_star,\
                                             alpha,\
                                             mode)
    
        subspace_errors_Hermite_q[k] = np.linalg.norm(U_true_list[k,:,:]-U_star, 'fro')/np.linalg.norm(U_true_list[k,:,:], 'fro')
    t_end = time.time()
    print('  ***   ')
    print('Hermite interpolation -q finished in ', t_end-t_start, 's')
    print('  ***   ')
    #
    ##
    ###
    ##
    #
    mode = 0  #q-centered calculation
    subspace_errors_Hermite_p = np.zeros((len(mu_range),))

    # preprocessing of sample data
    Deltas, Vs_shifted = sifs.Stiefel_Hermite_interp_pre(U_matrix_list,\
                                                         dU_matrix_list,\
                                                         mu_samples,\
                                                         alpha,\
                                                         mode)
    print('  ***   ')
    print('Start of Hermite interpolation p')
    print('  ***   ')
    t_start = time.time()
    # full interpolation
    for k in range(len(mu_range)):
        #trial point
        mu_star = mu_range[k]
        #print("Herm int at", mu_star)
        U_star = sifs.Stiefel_Hermite_interp(U_matrix_list,\
                                             dU_matrix_list,\
                                             Deltas,\
                                             Vs_shifted,\
                                             mu_samples,\
                                             mu_star,\
                                             alpha,\
                                             mode)
        subspace_errors_Hermite_p[k] = np.linalg.norm(U_true_list[k,:,:]-U_star, 'fro')/np.linalg.norm(U_true_list[k,:,:], 'fro')
    t_end = time.time()
    print('  ***   ')
    print('Hermite interpolation -p finished in ', t_end-t_start, 's')
    print('  ***   ')
# *****************************************************************************





print("max errors:")
if do_geo:
    print("Geodesic:", subspace_errors.max())
if do_tang_int:
    print("RBF full:", subspace_errors_RBF.max())
if do_Hermite:
    print("Hermite -q:", subspace_errors_Hermite_q.max())
    print("Hermite -p:", subspace_errors_Hermite_q.max())    
print("L2 errors:")
dt = abs(mu_range[1] - mu_range[0])
if do_geo:
    print("Geodesic:", math.sqrt(dt)*np.linalg.norm(subspace_errors))
if do_tang_int:
    print("RBF full:", math.sqrt(dt)*np.linalg.norm(subspace_errors_RBF))   
if do_Hermite:
    print("Hermite -q:", math.sqrt(dt)*np.linalg.norm(subspace_errors_Hermite_q))
    print("Hermite -p:", math.sqrt(dt)*np.linalg.norm(subspace_errors_Hermite_p))



# *****************************************************************************
#
# PLOT THE RESULTS
# 
# *****************************************************************************
do_plot = True
if do_plot:
    plt.rcParams.update({'font.size': 10})
    if do_geo:
        line_geod, = plt.plot(mu_range, subspace_errors, 'r-', linewidth=3, label = 'geodesic pw Pf')
    if do_tang_int:
        line_RBF,  = plt.plot(mu_range, subspace_errors_RBF, 'r-.', linewidth=3, label = 'RBF full Pf')
    if do_Hermite:
        line_Hermite,  = plt.plot(mu_range, subspace_errors_Hermite_q, 'r-', linewidth=1, label = 'Hermite pw, q-based')
        line_Hermite_p,  = plt.plot(mu_range, subspace_errors_Hermite_p, 'r:', linewidth=3, label = 'Hermite pw, p-based')

    plt.legend()
    plt.xlabel('mu')
    plt.ylabel('Errors')
    plt.show()

