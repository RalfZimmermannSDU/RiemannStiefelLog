# *****************************************************************************
# Interpolation of the Q-factor of a parametric QR decomposition
#
# This file implements the numerical experiment
# that features in Section 5.2 of
# 
# HERMITE INTERPOLATION AND DATA PROCESSING ERRORS ON RIEMANNIAN MATRIX MANIFOLDS
# SIAM J. SCI . COMPUT. Vol. 42, No. 5, pp. A2593--A2619
#
# MODIFICATIONS:
# Here: piecewise linear interpolation under various retractions
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
import Hermite_interp as HI

# set dimension
n = 50
p = 20



# QR or SVD
key = 'QR'

# metric
alpha = -0.5  # Euclidean metric

# create snapshots
np.random.seed(0)
Y0 = np.random.rand(n,p)
Y1 = 0.2*np.random.rand(n,p)
Y2 = 0.1*np.random.rand(n,p)
Y3 = 0.05*np.random.rand(n,p)
# samples:
# cheby-roots not important for local approaches but for the global 
# tang interp method
mu_samples = HI.ChebyRoots(0.5, 1.2, 3)
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
pw_lin_errors = np.zeros((len(mu_range),3)) # store errors for the three retractions
for retra in range(1,4):
    
    ref_coeffs = list()
    rbf_coeffs = list()
    # PREPROCESSING 
    Deltas = sifs.Stiefel_geodesic_interp_pre(U_matrix_list,\
                                              mu_samples,\
                                              alpha,\
                                              retra)

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
                                              alpha,
                                              retra)
    
        pw_lin_errors[k, retra-1] = np.linalg.norm(U_true_list[k,:,:]-U_star, 'fro')/np.linalg.norm(U_true_list[k,:,:], 'fro')
    t_end = time.time()
    print('  ***   ')
    print('Geodesic interpolation finished in ', t_end-t_start, 's')
    print('  ***   ')
    




print("max errors:")

print("PW linear, retra1 (Riemann     ):", pw_lin_errors[:,0].max())
print("PW linear, retra2 (polar factor):", pw_lin_errors[:,1].max())
print("PW linear, retra3 (polar light ):", pw_lin_errors[:,2].max())  

print("L2 relative errors:")
dt = abs(mu_range[1] - mu_range[0])

print("PW linear, retra1:", math.sqrt(dt)*np.linalg.norm(pw_lin_errors[:,0]))
print("PW linear, retra2:", math.sqrt(dt)*np.linalg.norm(pw_lin_errors[:,1]))
print("PW linear, retra3:", math.sqrt(dt)*np.linalg.norm(pw_lin_errors[:,2]))





# *****************************************************************************
#
# PLOT THE RESULTS
# 
# *****************************************************************************
do_plot = True
if do_plot:
    plt.rcParams.update({'font.size': 30})
  
    line_RBF1,  = plt.plot(mu_range, pw_lin_errors[:,0], 'r-.', linewidth=3, label = 'PW lin, retra1:Riemann')
    line_RBF2,  = plt.plot(mu_range, pw_lin_errors[:,1], 'k-.', linewidth=3, label = 'PW lin, retra2:PF')
    line_RBF3,  = plt.plot(mu_range, pw_lin_errors[:,2], 'b-.', linewidth=3, label = 'PW lin, retra3:PL')
    #line_pts, = plt.plot(mu_samples, np.zeros((len(mu_samples),)),  'bo', linewidth=3, label = 'fk')
    plt.legend()
    plt.xlabel('mu')
    plt.ylabel('Errors')
    plt.show()

