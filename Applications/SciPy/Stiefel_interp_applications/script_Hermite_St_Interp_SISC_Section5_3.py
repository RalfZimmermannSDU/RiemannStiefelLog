# *****************************************************************************
# Interpolation of an SVD decomposition
#
# This file implements the numerical experiment
# that features in Section 5.3 of
# 
# HERMITE INTERPOLATION AND DATA PROCESSING ERRORS ON RIEMANNIAN MATRIX MANIFOLDS
# SIAM J. SCI . COMPUT. Vol. 42, No. 5, pp. A2593--A2619
#
# It reproduces Figures 4 and 6 from this paper.
#
# @author: R. Zimmermann
# *****************************************************************************

import numpy as np
import math

import time

import matplotlib.pyplot as plt

import sys

sys.path.append('../../../Stiefel_log_general_metric/SciPy/')
sys.path.append('../Stiefel_interp/')
sys.path.append('../General_interp_tools/')

import snapshot_analytic_mat    as snam
import Stiefel_interp_funcs     as sifs
import Stiefel_Exp_Log          as StEL
import Stiefel_Aux              as StAux
import RBF_interp as RBF
import Hermite_interp as HI

# set dimension
n = 10000
p = 10
m = 300
# choose RBF function
#RBF_fun = RBF.Invmultiquad_rbf
#RBF_fun = RBF.Multiquad_rbf
#RBF_fun = RBF.TPS_rbf
#RBF_fun = RBF.Cubic_rbf
#RBF_fun = RBF.Invquad_rbf
RBF_fun = RBF.Lin_rbf
#RBF_fun = RBF.Gauss_rbf

#print( "")
#print( "**********************************")
#print( "Which experiments are to be performed?")
in_geo      = 1#input('      Geodesic interpolation 0/1?: ')
in_tang_int = 0#input(' Tangent space interpolation 0/1?: ')
in_Hermite  = 1#input(' Cubic Hermite interpolation 0/1?: ')

#covert input string from keyboard to integers
do_geo      = int(in_geo)
do_tang_int = int(in_tang_int)
do_Hermite  = int(in_Hermite)

# QR or SVD
key = 'SVD'
#CHECK, IF SVD IS ANALYTIC
check_analytic = 1

#metric paramter
alpha = 0.0


# create snapshots
np.random.seed(0)
Y0  = np.random.rand(n,p)
Y1  = 0.5*np.random.rand(n,p)
Y2  = 0.5*np.random.rand(n,p)
Y3  = 0.5*np.random.rand(n,p)
Z0  = np.random.rand(p,m)
Z1  = 0.5*np.random.rand(p,m)
Z2  = 0.5*np.random.rand(p,m)
# samples:
# cheby-roots not important for local approaches but for the global 
# tang interp method
mu_samples = HI.ChebyRoots(0.0, 0.5, 2)
mu_range   = np.linspace(mu_samples[0], mu_samples[-1], 10)

# create snapshot matrices of size n x p
U_matrix_list  = np.zeros((len(mu_samples), n,p))
dU_matrix_list = np.zeros((len(mu_samples), n,p))
Sigma_list     = np.zeros((len(mu_samples), p))
dSigma_list    = np.zeros((len(mu_samples), p))
V_matrix_list  = np.zeros((len(mu_samples), m,p))
dV_matrix_list = np.zeros((len(mu_samples), m,p))

print("Compute sample data")
for j in range(len(mu_samples)):

    U_matrix_list[j,:,:], dU_matrix_list[j,:,:], \
    Sigma_list[j,:], dSigma_list[j,:], \
    V_matrix_list[j,:,:], dV_matrix_list[j,:,:]  = \
                        snam.snapshot_analytic_lowrank(mu_samples[j], Y0,Y1,Y2,Y3, Z0,Z1,Z2, 1, key)


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
V_true_list = np.zeros((len(mu_range), m, p))

dU_true_list = np.zeros((len(mu_range), n, p))
dS_true_list = np.zeros((len(mu_range), p))
dV_true_list = np.zeros((len(mu_range), m, p))

print("Compute reference data")
for k in range(len(mu_range)):
    #trial point
    mu_star = mu_range[k]   
    # for comparison: true matrix
    U,dU, S, dS, V, dV = snam.snapshot_analytic_lowrank(mu_star,\
                                                        Y0,Y1,Y2,Y3,\
                                                        Z0,Z1,Z2,\
                                                        1, key)
    U_true_list[k,:,:] = U
    S_true_list[k,:]   = S
    V_true_list[k,:,:] = V
    dU_true_list[k,:,:]= dU
    dS_true_list[k,:]  = dS
    dV_true_list[k,:,:]= dV


# *****************************************************************************
# perform interpolation:
# *****************************************************************************


# *****************************************************************************
# FIRST EXPERIMENT: no derivatives
# linear interpolation for "all" mu in sample range
# works in only in 1D 
# *****************************************************************************
if do_geo:
    Total_geo_errors = np.zeros((len(mu_range),)) 
    SVD_geo_errors = np.zeros((len(mu_range),3))
    
    # PREPROCESSING 
    Deltas_U = sifs.Stiefel_geodesic_interp_pre(U_matrix_list, mu_samples, alpha)

    Deltas_V = sifs.Stiefel_geodesic_interp_pre(V_matrix_list, mu_samples, alpha)

    print('  ***   ')
    print('Start of geodesic interpolation')
    print('  ***   ')
    t_start = time.time()
    
    ref_coeffs = list()
    rbf_coeffs = list()
    # remark: only two samples, same tangent vector 
    # Delta =  Stiefel_log(U1, U2) throughout
    # code could be optimized!
    for k in range(len(mu_range)):
        #trial point
        mu_star = mu_range[k]
        print("geo int at", mu_star)
        U_star = sifs.Stiefel_geodesic_interp(U_matrix_list,\
                                              Deltas_U,\
                                              mu_samples,\
                                              mu_star,\
                                              alpha)
        V_star = sifs.Stiefel_geodesic_interp(V_matrix_list,\
                                              Deltas_V,\
                                              mu_samples,\
                                              mu_star,\
                                              alpha)
              
        # linear interp for singular values
        # assumption: samples is an ordered list of mu-values
        aux = abs(mu_samples - mu_star)
        index = np.argmin(aux)
        if (mu_star < mu_samples[index]) or abs(mu_star - mu_samples[-1])<1.0e-15:
            pos = index-1
        else:
            pos = index
            # interval bounds
        mu0 = mu_samples[pos]
        mu1 = mu_samples[pos+1]
        lin_factor = (mu_star - mu0)/(mu1- mu0)
        S_star = Sigma_list[pos, :] + lin_factor * (Sigma_list[pos+1, :] - Sigma_list[pos, :])
   
        #----------------------------------------------------------------------     
        # store errors
        #----------------------------------------------------------------------
        #errors in U
        SVD_geo_errors[k,0] = np.linalg.norm(U_true_list[k,:,:]-U_star, 'fro')/np.linalg.norm(U_true_list[k,:,:], 'fro')
       #errors in S
        SVD_geo_errors[k,1] = np.linalg.norm(S_true_list[k,:]-S_star)/np.linalg.norm(S_true_list[k,:])
        #errors in V
        SVD_geo_errors[k,2] = np.linalg.norm(V_true_list[k,:,:]-V_star, 'fro')/np.linalg.norm(V_true_list[k,:,:], 'fro')
        
        # total reconstruction error
        Recon_SVD = np.dot(U_star, np.dot(np.diag(S_star) , V_star.T))
        True_SVD  = np.dot(U_true_list[k,:,:],\
                              np.dot(np.diag(S_true_list[k,:]) , V_true_list[k,:,:].T))
        
        Total_geo_errors[k] = np.linalg.norm(Recon_SVD-True_SVD, 'fro')/np.linalg.norm(True_SVD, 'fro')
    t_end = time.time()
    print('  ***   ')
    print('Geodesic interpolation finished in ', t_end-t_start, 's')
    print('  ***   ')
    

# *****************************************************************************
# Second experiemtn: Quasi-Cubic Hermite Interpolation
# *****************************************************************************
if do_Hermite:
    Total_Hermite_errors = np.zeros((len(mu_range),))
    SVD_Hermite_errors   = np.zeros((len(mu_range),5))
        # first three entries: errors in U, S, V
        # entries 4 and 5: manifold errors, tangent space errors
        
    # preprocessing of sample data
    Deltas_U, Vs_shifted_U = sifs.Stiefel_Hermite_interp_pre(U_matrix_list,\
                                                             dU_matrix_list,\
                                                             mu_samples,\
                                                             alpha)
    Deltas_V, Vs_shifted_V = sifs.Stiefel_Hermite_interp_pre(V_matrix_list,\
                                                             dV_matrix_list,\
                                                             mu_samples,\
                                                             alpha)
    print('  ***   ')
    print('Start of Hermite interpolation')
    print('  ***   ')
    t_start = time.time()  
    
    S_star_Herm = np.zeros((len(mu_range), p))
    # full interpolation
    for k in range(len(mu_range)):
        #trial point
        mu_star = mu_range[k]
        print("Herm int at", mu_star)
        U_star = sifs.Stiefel_Hermite_interp(U_matrix_list,\
                                             dU_matrix_list,\
                                             Deltas_U,\
                                             Vs_shifted_U,\
                                             mu_samples,\
                                             mu_star,\
                                             alpha)
        V_star = sifs.Stiefel_Hermite_interp(V_matrix_list,\
                                             dV_matrix_list,\
                                             Deltas_V,\
                                             Vs_shifted_V,\
                                             mu_samples,\
                                             mu_star,\
                                             alpha)
        S_star = HI.Hermite_vec_interp(Sigma_list,\
                                       dSigma_list,\
                                       mu_samples,\
                                       mu_star)
        #  S_star_Herm is only used to do plots of the singular values
        S_star_Herm[k,:] = S_star
        
        #----------------------------------------------------------------------     
        # store errors
        #----------------------------------------------------------------------
        #errors in U
        SVD_Hermite_errors[k,0] = np.linalg.norm(U_true_list[k,:,:]-U_star, 'fro')/np.linalg.norm(U_true_list[k,:,:], 'fro')
        #errors in S
        SVD_Hermite_errors[k,1] = np.linalg.norm(S_true_list[k,:]-S_star)/np.linalg.norm(S_true_list[k,:])
        #errors in V
        SVD_Hermite_errors[k,2] = np.linalg.norm(V_true_list[k,:,:]-V_star, 'fro')/np.linalg.norm(V_true_list[k,:,:], 'fro')
        
        #errors tangent space vs manifold
        if 1:
            SVD_Hermite_errors[k,3] = StEL.distStiefel(U_star, U_true_list[k,:,:])
        
            Delta_star, iter_count =  StEL.Stiefel_Log(U_matrix_list[1,:,:], U_star, 1.0e-14)
            Delta_true, iter_count =  StEL.Stiefel_Log(U_matrix_list[1,:,:], U_true_list[k,:,:], 1.0e-14)
            D_error = Delta_star - Delta_true        
            SVD_Hermite_errors[k,4] = math.sqrt(StAux.alphaMetric(D_error, D_error, U_matrix_list[1,:,:]))
        
        # total reconstruction error        
        Recon_SVD = np.dot(U_star, np.dot(np.diag(S_star) , V_star.T))
        
        True_SVD  = np.dot(U_true_list[k,:,:],\
                           np.dot(np.diag(S_true_list[k,:]) , V_true_list[k,:,:].T))
        
        #subspace_errors_Hermite[k] = np.linalg.norm(U_true_list[k,:,:]-U_star, 'fro')/np.linalg.norm(U_true_list[k,:,:], 'fro')
        Total_Hermite_errors[k] = np.linalg.norm(Recon_SVD-True_SVD, 'fro')/np.linalg.norm(True_SVD, 'fro')
    
    t_end = time.time()
    print('  ***   ')
    print('Hermite interpolation finished in ', t_end-t_start, 's')
    print('  ***   ')
# *****************************************************************************




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
    
    Total_RBF_errors = np.zeros((len(mu_range),))
    SVD_RBF_errors = np.zeros((len(mu_range),3))

    S_star_RBF = np.zeros((len(mu_range), p))
    # PREPROCESSING: map all sample locations to tangent space
    Deltas_U = sifs.Stiefel_RBF_tang_interp_pre(U_matrix_list,\
                                                sample_sites,\
                                                alpha)
    
    Deltas_V = sifs.Stiefel_RBF_tang_interp_pre(V_matrix_list,\
                                                sample_sites,\
                                                alpha)

    # inverse of correlation matrix stays fixed
    Rinv = RBF.RBF_invmat(sample_sites, RBF_fun)
    print('  ***   ')
    print('Start of RBF space interpolation')
    print('  ***   ')
    t_start = time.time()
    # full interpolation
    for k in range(len(mu_range)):
        #trial point
        mu_star = mu_range[k]
        #print("tang int at", mu_star)
        U_star = sifs.Stiefel_RBF_tang_interp(U_matrix_list,\
                                              sample_sites,\
                                              Deltas_U,\
                                              mu_star,\
                                              Rinv,\
                                              RBF_fun,\
                                              alpha)
        
        V_star = sifs.Stiefel_RBF_tang_interp(V_matrix_list,\
                                              sample_sites,\
                                              Deltas_V,\
                                              mu_star,\
                                              Rinv,\
                                              RBF_fun,\
                                              alpha)
        # 2.) build RBF interpolator
        RBF_coeffs = RBF.RBF_mat_interp(mu_star, Rinv, sample_sites, RBF_fun)
    
        S_star = np.zeros((p,))
        for i in range(len(mu_samples)):
            S_star += RBF_coeffs[i]*Sigma_list[i,:]
        #  S_star_Herm is only used to do plots of the singular values
        S_star_RBF[k,:] = S_star
        #----------------------------------------------------------------------     
        # store errors
        #----------------------------------------------------------------------
        #errors in U
        SVD_RBF_errors[k,0] = np.linalg.norm(U_true_list[k,:,:]-U_star, 'fro')/np.linalg.norm(U_true_list[k,:,:], 'fro')
        #errors in S
        SVD_RBF_errors[k,1] = np.linalg.norm(S_true_list[k,:]-S_star)/np.linalg.norm(S_true_list[k,:])
        #errors in V
        SVD_RBF_errors[k,2] = np.linalg.norm(V_true_list[k,:,:]-V_star, 'fro')/np.linalg.norm(V_true_list[k,:,:], 'fro')
 
        # total reconstruction error        
        Recon_SVD = np.dot(U_star, np.dot(np.diag(S_star) , V_star.T))
        
        True_SVD  = np.dot(U_true_list[k,:,:],\
                           np.dot(np.diag(S_true_list[k,:]) , V_true_list[k,:,:].T))
        
        #subspace_errors_Hermite[k] = np.linalg.norm(U_true_list[k,:,:]-U_star, 'fro')/np.linalg.norm(U_true_list[k,:,:], 'fro')
        Total_RBF_errors[k] = np.linalg.norm(Recon_SVD-True_SVD, 'fro')/np.linalg.norm(True_SVD, 'fro')
    
    t_end = time.time()
    print('  ***   ')
    print('RBF interpolation finished in ', t_end-t_start, 's')
    print('  ***   ')
# *****************************************************************************
      
# *****************************************************************************





print("max relative errors:")
if do_geo:
    print("Geodesic:", Total_geo_errors.max())
if do_Hermite:
    print("Hermite :", Total_Hermite_errors.max())
print("L2 relative errors:")
dt = abs(mu_range[1] - mu_range[0])
if do_geo:
    print("Geodesic:", math.sqrt(dt)*np.linalg.norm(Total_geo_errors))
if do_Hermite:
    print("Hermite :", math.sqrt(dt)*np.linalg.norm(Total_Hermite_errors))




if check_analytic: # check if SVD is analytic in mu
    line_sigma_pm2,  = plt.plot(mu_range, S_true_list[:,p-3], 'k-', label = 'sigma p-2')
    line_sigma_pen,  = plt.plot(mu_range, S_true_list[:,p-2], 'b-', label = 'sigma p-1')
    line_sigma_min,  = plt.plot(mu_range, S_true_list[:,p-1], 'r-', label = 'sigma p')
    
    line_singvec1,  = plt.plot(mu_range, S_star_Herm[:,p-3], 'k--', label = 'sigma p-2 interp')
    line_singvec2,  = plt.plot(mu_range, S_star_Herm[:,p-2], 'b--', label = 'sigma p-1 interp')
    line_singvec2,  = plt.plot(mu_range, S_star_Herm[:,p-1], 'r--', label = 'sigma p interp')

    plt.legend()
    plt.xlabel('mu')
    plt.ylabel('sigma')
    plt.show()


# *****************************************************************************
#
# PLOT THE RESULTS
# 
# *****************************************************************************
do_plot1 = True
if do_plot1:
    plt.rcParams.update({'font.size': 30})
    if do_geo:
        line_geod, = plt.plot(mu_range, Total_geo_errors, 'k-', linewidth=3, label = 'geodesic pw')
    if do_tang_int:
        line_RBF, = plt.plot(mu_range, Total_RBF_errors, 'k-.', linewidth=3, label = 'RBF full')
    if do_Hermite:
        line_geod, = plt.plot(mu_range, Total_Hermite_errors, 'k--', linewidth=3, label = 'cubic Hermite')
    plt.legend()
    plt.xlabel('mu')
    plt.ylabel('Errors')
    plt.show()

do_plot2 = True
if do_plot2:
    plt.rcParams.update({'font.size': 30})
    if do_geo:
        line_geo_U, = plt.plot(mu_range, SVD_geo_errors[:,0], 'b-', linewidth=3, label = 'U geo')
        line_geo_S, = plt.plot(mu_range, SVD_geo_errors[:,1], 'k-', linewidth=3, label = 'S geo')
        line_geo_V, = plt.plot(mu_range, SVD_geo_errors[:,2], 'r-', linewidth=3, label = 'V geo')      
    if do_Hermite:
        line_herm_U, = plt.plot(mu_range, SVD_Hermite_errors[:,0], 'b--', linewidth=3, label = 'U Hermite')
        line_herm_S, = plt.plot(mu_range, SVD_Hermite_errors[:,1], 'k--', linewidth=3, label = 'S Hermite')
        line_herm_V, = plt.plot(mu_range, SVD_Hermite_errors[:,2], 'r--', linewidth=3, label = 'V Hermite')           
    plt.legend()
    plt.xlabel('mu')
    plt.ylabel('Errors')
    plt.show()


do_plot3 = True
if do_plot3:
    plt.rcParams.update({'font.size': 30})     
    if do_Hermite:
        line_herm_Man, = plt.plot(mu_range, SVD_Hermite_errors[:,3], 'k--', linewidth=3, label = 'Man error')
        line_herm_Tan, = plt.plot(mu_range, SVD_Hermite_errors[:,4], 'k-', linewidth=2, label = 'Tan error')         
    plt.legend()
    plt.xlabel('mu')
    plt.ylabel('Errors')
    plt.show()
    
