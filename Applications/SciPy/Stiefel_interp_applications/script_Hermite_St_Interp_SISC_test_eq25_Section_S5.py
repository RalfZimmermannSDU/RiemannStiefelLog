# *****************************************************************************
# Code validation and numerical accuracy tests
#
# This file eventually implements the numerical experiment
# that features in Section S5 of the supplements of
# 
# HERMITE INTERPOLATION AND DATA PROCESSING ERRORS ON RIEMANNIAN MATRIX MANIFOLDS
# SIAM J. SCI . COMPUT. Vol. 42, No. 5, pp. A2593--A2619
#
# It is associated with eq. (16) and (25) and produces a plot similar
# to Figures S3, S4, S5.
# Due to improvements in the Riemannian log algorithm the results differ
# slightly from the ones in the published paper.
#
# @author: R. Zimmermann
# *****************************************************************************

import scipy as sc
import numpy as np
import math
from scipy import random
import matplotlib.pyplot as plt

import sys

# import local modules

sys.path.append( '../../../Stiefel_log_general_metric/SciPy/')
sys.path.append( '../Stiefel_interp/')
sys.path.append( '../General_interp_tools/')

import Stiefel_Exp_Log          as StEL
import Stiefel_Aux              as StAux
import snapshot_analytic_mat    as snam
import Stiefel_diff_tools       as sdiffs
import Stiefel_tests



# set dimension
n = 1000
p = 6
# choose metric alpha
alpha = -0.5

# create snapshots
# t-samples:
t_range = np.linspace(1.0,4.0, p)

# mu-samples
mu_samples = np.array([0.9, 1.4, 1.9])

# create some analytic snapshot matrices of size n x p
U_matrix_list = np.zeros((len(mu_samples), n,p))

for j in range(len(mu_samples)):
    U_matrix_list[j,:,:], Sigma, VT = \
    snam.snapshot_analytic_mat2(t_range, mu_samples[j], n, np.array([[None]]), 0, 'SVD')
                              #(t_range,          mu, dim,   RefU,    comp_deriv,  key)
 
# *****************************************************************************
# at this stage, each matrix
# Uj = U_matrix_list(j,:,:)
# is a  column-orthogonal matrix that
# depends smoothly on mu
# *****************************************************************************


# *****************************************************************************
# FIRST EXPERIMENT
# *****************************************************************************
print("")
print("***************************************************************")
print("Numerical validation in -script_test_Hermite_Stiefel_interp.py-")
print("***************************************************************")
print(" First EXPERIMENT")
print(" use two data points: U0,U1")
print(" Task: Derivative of Stiefel curve")
print(" * compute forward  geodesic gamma0 from U0 to U1: velocity: Delta01")
print(" * compute backward geodesic gamma1 from U1 to U0: velocity: Delta10")
print(" * check if derivative  gamma1'(1) matches -gamma0'(0)")
D1, D2 = Stiefel_tests.test_diff_Stiefel_curve(U_matrix_list[0,:,:],\
                                          U_matrix_list[2,:,:], 1, alpha)
                                          
print("")
print("***************************************************************")
print(" Second EXPERIMENT")
print(" Points U0, U1, U2")
print(" Velocities: ")
print("   V12 in T_U1 (start velocity of geodesic from U1 to U2)")
print("   V02 in T_U0 (start velocity of geodesic from U0 to U2)")
print("")
print(" interpolate and match gradients")
print(" use U1 as center to attach tangent space")
print("***************************************************************")
U0 = U_matrix_list[0,:,:] 
U1 = U_matrix_list[1,:,:] 
U2 = U_matrix_list[2,:,:] 
# step 1: from U1 to U0
# use this as location in tangent space
Delta10, iter_count =  StEL.Stiefel_Log(U1, U0, 1.0e-13, alpha)
# step 2: from U0 to U2
# use this as a prescribed velocity vector at U0
V02, iter_count =  StEL.Stiefel_Log(U0, U_matrix_list[2,:,:], 1.0e-13, alpha)
# step 3: from U1 to U2 
# use this as a prescribed velocity vector at U1
V12, iter_count =  StEL.Stiefel_Log(U1, U2, 1.0e-13, alpha)
#******************************************************************************
# major step: translate V02 to a velocity in T_U1
#
# compute V0 = d/dt|t=0   (Log_U1 o Exp_U0)(t*V02)
# use central difference
#******************************************************************************

# FD step size
h = 1.0e-4
T = StEL.Stiefel_Exp(U0, h*V02, alpha)
fplus, iter_count =  StEL.Stiefel_Log(U1, T, 1.0e-13, alpha)
T2 = StEL.Stiefel_Exp(U0, -h*V02, alpha)
fminus, iter_count =  StEL.Stiefel_Log(U1, T2, 1.0e-13, alpha)
V0 = (1.0/(2*h))*fplus - (1.0/(2*h))*fminus

# alternative: finite difference
#V0 = (1.0/h)*fplus - (1.0/h)*Delta10

#******************************************************************************
#  validate the implementation
#******************************************************************************
dSt_Exp = sdiffs.Stiefel_diff_exp(U1, Delta10, V0, alpha)

Atest = np.dot(dSt_Exp.transpose(), U0)

print("norm check: Is dExp in tangent space?",\
 sc.linalg.norm(Atest+Atest.T, 'fro'))
 
print("norm check: Was dExp recovered?",\
 sc.linalg.norm(dSt_Exp-V02, 'fro')/sc.linalg.norm(V02, 'fro'))

print('distance', StEL.distStiefel(U0, U1))


#******************************************************************************
# Repeat in  loop over increasing distance
#******************************************************************************
print("")
print("***************************************************************")
print(" Third EXPERIMENT")
print(" Associated with eqs. (16), (25) and supplemets section S5 of ")
print(" HERMITE INTERPOLATION AND DATA PROCESSING ERRORS ON RIEMANNIAN MATRIX MANIFOLDS")
print(" SIAM J. SCI . COMPUT. Vol. 42, No. 5, pp. A2593--A2619")

print("***************************************************************")


n = 1000
p = 5
dist = 0.3

# create a location Up and tangent vector V_p (output U1p is not used)
Up, U1p, V_p = StEL.create_random_Stiefel_data(n, p, dist)

# create a second  tangent vector Delta
A = random.rand(p,p)
A = A-A.transpose()   # "random" p-by-p skew symmetric matrix
T = random.rand(n,p)
Gap_delta = np.dot(Up,A)+ T-np.dot(Up,np.dot(Up.transpose(),T))
#normalize Gap_delta w.r.t. the canonical metric, default alpha = 0
norm_Delta = math.sqrt(StAux.alphaMetric(Gap_delta, Gap_delta, Up))
Gap_delta = (1.0/norm_Delta)*Gap_delta
print('base gap size', math.sqrt(StAux.alphaMetric(Gap_delta, Gap_delta, Up)))


# create range of distance scaling factors
dist_range = np.linspace(0.01, (0.9*sc.pi), 10)
# FD step size
h = 1.0e-4
# accuracy for log-computation
tau = 1.0e-13
# keep track of errors
errors_FD = np.zeros((len(dist_range),))

for k in range(len(dist_range)):
    
    # in every loop iteration, U1 moves further away from Up
    U1 = StEL.Stiefel_Exp(Up, dist_range[k]*Gap_delta)
    
    # compute Riemannian log from basis U1
    Delta10, iter_count =  StEL.Stiefel_Log(U1, Up, tau)
    T = StEL.Stiefel_Exp(Up, h*V_p)
    fplus, iter_count =  StEL.Stiefel_Log(U1, T, tau)
    
    T2 = StEL.Stiefel_Exp(Up, -h*V_p)
    fminus, iter_count =  StEL.Stiefel_Log(U1, T2, tau)
    V0 = (1.0/(2*h))*fplus - (1.0/(2*h))*fminus

    #**************************************************************************
    #  validate the implementation
    #**************************************************************************
    dSt_Exp = sdiffs.Stiefel_diff_exp(U1, Delta10, V0)

    Atest = np.dot(dSt_Exp.transpose(), Up)

    print("norm check: Is dExp in tangent space?",\
          sc.linalg.norm(Atest+Atest.T, 'fro'))
 
    errors_FD[k] =  sc.linalg.norm(dSt_Exp-V_p, 'fro')/sc.linalg.norm(V_p, 'fro')
    print("norm check: Was dExp recovered?",\
          sc.linalg.norm(dSt_Exp-V_p, 'fro')/sc.linalg.norm(V_p, 'fro'))

# plot the results
plt.rcParams.update({'font.size': 20})
line_err, = plt.plot(dist_range, errors_FD, 'k-', linewidth=3, label = 'error')
plt.legend()
plt.yscale('log')
plt.xlabel('dist(U0,U1)')
plt.ylabel('Errors according to eq. (25)')
plt.title('Check accuracy of FD approximation vs. Riemannian distance')
plt.show()

 