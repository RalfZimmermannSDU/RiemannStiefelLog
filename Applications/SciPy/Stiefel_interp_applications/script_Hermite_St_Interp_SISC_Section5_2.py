# *****************************************************************************
# Interpolation of the Q-factor of a parametric QR decomposition
#
# This file implements the numerical experiment
# that features in Section 5.2 of
# 
# HERMITE INTERPOLATION AND DATA PROCESSING ERRORS ON RIEMANNIAN MATRIX MANIFOLDS
# SIAM J. SCI . COMPUT. Vol. 42, No. 5, pp. A2593--A2619
#
# The choice "RBF_fun = RBF.Invquad_rbf" reproduces Figure 4 from this paper.
#
# @author: R. Zimmermann
# *****************************************************************************

import numpy as np
import math
import matplotlib.pyplot as plt
import time
import sys

sys.path.append('../../../Stiefel_log_general_metric/SciPy/')
sys.path.append('../Stiefel_interp/')
sys.path.append('../General_interp_tools/')

import snapshot_analytic_mat    as snam
import Stiefel_interp_funcs     as sifs
import RBF_interp as RBF
import Hermite_interp as HI

# set dimension
n = 500
p = 10
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

do_geo      = int(in_geo)
do_tang_int = int(in_tang_int)
do_Hermite  = int(in_Hermite)

# QR or SVD
key = 'QR'

# metric
alpha = 0.0

# create snapshots
np.random.seed(0)
Y0 = np.random.rand(n,p)
Y1 = 0.5*np.random.rand(n,p)
Y2 = 0.5*np.random.rand(n,p)
Y3 = 0.2*np.random.rand(n,p)
# samples:
# cheby-roots not important for local approaches but for the global 
# tang interp method
mu_samples = HI.ChebyRoots(-1.1, 1.1, 6)
print('Cheby samples:', mu_samples)
mu_range = np.linspace(mu_samples[0], mu_samples[-1], 101)

# create some analytic snapshot matrices of size n x p
U_matrix_list  = np.zeros((len(mu_samples), n,p))
dU_matrix_list = np.zeros((len(mu_samples), n,p))
for j in range(len(mu_samples)):
    U_matrix_list[j,:,:], dU_matrix_list[j,:,:] = \
                        snam.snapshot_analytic_mat3(mu_samples[j], Y0,Y1,Y2,Y3, 1, key)
 
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

for k in range(len(mu_range)):
    #trial point
    mu_star = mu_range[k]   
    # for comparison: true matrix
    U = snam.snapshot_analytic_mat3(mu_star, Y0,Y1,Y2,Y3,0, key)
    U_true_list[k,:,:] = U


# *****************************************************************************
# perform interpolation:
# *****************************************************************************


# *****************************************************************************
# FIRST EXPERIMENT
# linear interpolation for "all" mu in sample range
# works in only in 1D 
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
        U_star = sifs.Stiefel_geodesic_interp(U_matrix_list, Deltas,\
                                              mu_samples, mu_star, alpha)
    
        subspace_errors[k] = np.linalg.norm(U_true_list[k,:,:]-U_star, 'fro')/np.linalg.norm(U_true_list[k,:,:], 'fro')
    t_end = time.time()
    print('  ***   ')
    print('Geodesic interpolation finished in ', t_end-t_start, 's')
    print('  ***   ')
    

# *****************************************************************************
# Next EXPERIMENT
# RBF interpolation in tangent space
# * map all data to single tangent space
# * perform RBF interpolation in tangent space
# 
# *****************************************************************************
if do_tang_int:
    sample_sites = np.zeros((1, len(mu_samples)))
    sample_sites[0,:] = mu_samples
    subspace_errors_RBF = np.zeros((len(mu_range),))

    # map all sample locations to tangent space
    Deltas = sifs.Stiefel_RBF_tang_interp_pre(U_matrix_list,\
                                              sample_sites, alpha)
    
        # inverse of correlation matrix stays fixed
    Rinv = RBF.RBF_invmat(sample_sites, RBF_fun)
    # full interpolation
    print('  ***   ')
    print('Start of tangent space interpolation')
    print('  ***   ')
    t_start = time.time()
    for k in range(len(mu_range)):
        #trial point
        mu_star = mu_range[k]
        #print("tang int at", mu_star)
        U_star = sifs.Stiefel_RBF_tang_interp(U_matrix_list,\
                                              sample_sites,\
                                              Deltas,\
                                              mu_star,\
                                              Rinv,\
                                              RBF_fun, alpha)
    
        subspace_errors_RBF[k] = np.linalg.norm(U_true_list[k,:,:]-U_star, 'fro')/np.linalg.norm(U_true_list[k,:,:], 'fro')
    t_end = time.time()
    print('  ***   ')
    print('Tangent space interpolation finished in ', t_end-t_start, 's')
    print('  ***   ')

# *****************************************************************************
# Next EXPERIMENT
# Quasi-Cubic Hermite Interpolation
# 
# *****************************************************************************
if do_Hermite:
    mode = 1 # q-centered
    subspace_errors_Hermite_q = np.zeros((len(mu_range),))
    # preprocessing of sample data
    Deltas, Vs_shifted = sifs.Stiefel_Hermite_interp_pre(U_matrix_list,\
                                                         dU_matrix_list,\
                                                         mu_samples,\
                                                         alpha,\
                                                         mode)
    print('  ***   ')
    print('Start of Hermite interpolation q')
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
    
    
    mode = 0 # p-centered
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
    for k in range(len(mu_range)-1):
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
    print("Hermite q:", subspace_errors_Hermite_q.max())
    print("Hermite p:", subspace_errors_Hermite_p.max())

print("L2 relative errors:")
dt = abs(mu_range[1] - mu_range[0])
if do_geo:
    print("Geodesic:", math.sqrt(dt)*np.linalg.norm(subspace_errors))
if do_tang_int:
    print("Tan space:", math.sqrt(dt)*np.linalg.norm(subspace_errors_RBF))
if do_Hermite:
    print("Hermite q:", math.sqrt(dt)*np.linalg.norm(subspace_errors_Hermite_q))
    print("Hermite p:", math.sqrt(dt)*np.linalg.norm(subspace_errors_Hermite_p))

 




# *****************************************************************************
#
# PLOT THE RESULTS
# 
# *****************************************************************************
do_plot = True
if do_plot:
    plt.rcParams.update({'font.size': 30})
    if do_geo:
        line_geod, = plt.plot(mu_range, subspace_errors, 'k-', linewidth=3, label = 'geodesic pw')
    if do_tang_int:
        line_RBF,  = plt.plot(mu_range, subspace_errors_RBF, 'r-.', linewidth=3, label = 'RBF full')
    if do_Hermite:
        line_Hermite_q,  = plt.plot(mu_range, subspace_errors_Hermite_q, 'r-', linewidth=1, label = 'Hermite pw, q-based')
        line_Hermite_p,  = plt.plot(mu_range, subspace_errors_Hermite_q, 'r:', linewidth=3, label = 'Hermite pw, p-based')
  
    #line_pts, = plt.plot(mu_samples, np.zeros((len(mu_samples),)),  'bo', linewidth=3, label = 'fk')
    plt.legend()
    plt.xlabel('mu')
    plt.ylabel('Errors')
    plt.show()

