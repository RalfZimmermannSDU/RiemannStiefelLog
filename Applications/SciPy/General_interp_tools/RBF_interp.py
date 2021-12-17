import numpy as np
import math

from scipy import interpolate
from scipy import linalg


import matplotlib.pyplot as plt


###############################################################################
# FUNCTION DECLARATION
###############################################################################


#------------------------------------------------------------------------------
# compute coeffs via RBF interpolation
# This function returns the value of the basis coefficients 
# at the parameter point x_star 
# !!USES SCIPY FUNCTION!!
#------------------------------------------------------------------------------
def RBF_coeffs(x_star, sample_sites, nr_paras, RBF_type):
    
    if nr_paras > 1:
        nr_of_basis_vectors = sample_sites.shape[1]
    else:
        nr_of_basis_vectors = sample_sites.size

    coeffs = np.zeros(nr_of_basis_vectors)
    for l in range(nr_of_basis_vectors):
        #create lth unit vector as sample values
        sample_values = np.zeros(nr_of_basis_vectors)
        sample_values[l] = 1.0
        #initialize interpolator
        if nr_paras > 1:
            rbf_interpolator = interpolate.Rbf(sample_sites[0,:],\
                                               sample_sites[1,:],\
                                               sample_values,\
                                               function = RBF_type)
            coeffs[l] = rbf_interpolator(x_star[0], x_star[1])
        else:
            rbf_interpolator = interpolate.Rbf(sample_sites,\
                                               sample_values,\
                                               function = RBF_type)
            coeffs[l] = rbf_interpolator(x_star[0])
    return coeffs
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# This function returns the inverse of the sample correlation matrix
# The columns of Rinv are the solutions to
#   c = R^-1 ej
# which are the coefficient for matrix interpolation problems
#
# inputs: 
# sample_sites = matrix of sampled locations, column-wise
#                |x^1 , x^2, ... , x^k|
# RBF_fun      = selected RBF function
# outputs: 
# Rinv = RBF^{-1}
#
#
#------------------------------------------------------------------------------
def RBF_invmat(sample_sites, RBF_fun):
    
    # get number of samples
    nr_of_basis_vectors = sample_sites.shape[1]

    # initialize corr matrix
    corr_matrix = np.zeros([nr_of_basis_vectors, nr_of_basis_vectors])
    
    # build correlation matrix
    for i in range(nr_of_basis_vectors):
        corr_matrix[i,i] = RBF_fun(0.0)          # RBF
        for j in range(i+1, nr_of_basis_vectors):
            r_ij = np.linalg.norm(sample_sites[:,i] - sample_sites[:,j], 2)
            corr_matrix[i,j] = RBF_fun(r_ij)     #radial basis function
            corr_matrix[j,i] = corr_matrix[i,j]
    
    Rinv = linalg.inv(corr_matrix, overwrite_a = True)    
    
    return Rinv
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# This function returns the coefficient vector
# for matrix interpolation problems
#
# iputs: 
#  xstar       = vector-valued parameter location
#   Rinv       = inverse of correlation matrix
# sample_sites = matrix of sampled locations, column-wise
#                |x^1 , x^2, ... , x^k|
# RBF_fun      = selected RBF function
#
#------------------------------------------------------------------------------
def RBF_mat_interp(xstar, Rinv, sample_sites, RBF_fun):
    
    nr_samples = sample_sites.shape[1]

    r_vec = np.zeros((nr_samples,))
    # build correlation vector
    for i in range(nr_samples):
        r_i = np.linalg.norm(sample_sites[:,i] - xstar, 2)
        r_vec[i] =  RBF_fun(r_i)     #radial basis function
    RBF_coeff_vec = np.dot(Rinv, r_vec)
    return RBF_coeff_vec
#-----------------------------------------------------------------------------


#------------------------------------------------------------------------------
# plot RBF interpolation function (1D only)
# plot intyerpolator for first, middle and last coeff
# Convention: mu samples are listed column-wise
# => len(samples[0]) is the number of sample points
#------------------------------------------------------------------------------
def plot_RBF_interp(samples,\
                    RBF_fun):
    pos0 = int(samples.shape[1]/2)
    res = 101
    print("plot_RBF_interp: pos0 =:", pos0)

    Rinv = RBF_invmat(samples, RBF_fun)
    
    print("Condition number of RBF matrix: ", np.linalg.cond(Rinv) )

    mu_star_range = np.linspace(samples[0,0], samples[0,-1], res)
    RBF_plt_interp = np.zeros((res,3))
    
    # 2.) perform RBF interpolation
    for j in range(res):
        RBF_coeffs = RBF_mat_interp(mu_star_range[j], Rinv, samples, RBF_fun)
        RBF_plt_interp[j,0] = RBF_coeffs[0]
        RBF_plt_interp[j,1] = RBF_coeffs[pos0]       
        RBF_plt_interp[j,2] = RBF_coeffs[-1]     
    
    # plot RBF interpolator
    # sample values
    e0          = np.zeros((len(samples[0]),))
    e0[0]       = 1.0
    e_mid       = np.zeros((len(samples[0]),))
    e_mid[pos0] = 1.0
    e_end       = np.zeros((len(samples[0]),))
    e_end[-1]   = 1.0
    
    plt.rcParams.update({'font.size': 20})
    line_a0, = plt.plot(mu_star_range,  RBF_plt_interp[:,0],\
                        'k-', linewidth=3, label = 'RBFcoeff 0')
    line_a1, = plt.plot(mu_star_range,  RBF_plt_interp[:,1],\
                        'b-.', linewidth=3, label = 'RBFcoeff mid')
    line_a2, = plt.plot(mu_star_range,  RBF_plt_interp[:,2],\
                        'r--', linewidth=3, label = 'RBFcoeff end')
    line_b0, = plt.plot(samples[0], e0, 'ko', markersize=12)
    line_b1, = plt.plot(samples[0], e_mid, 'bo', markersize=12)   
    line_b2, = plt.plot(samples[0], e_end, 'ro', markersize=12)   
    plt.legend()
    plt.xlabel('mu')
    plt.ylabel('RBF(mu)')
    plt.show()
# END: plot RBF interpolation 
#------------------------------------------------------------------------------







#TPS radial basis function
def TPS_rbf(r):
    if abs(r) <1.0e-14:
        return 0.0
    else:
        return np.log(r)*r*r

# Linear radial basis function
def Lin_rbf(r):
    return r

# Cubic radial basis function
def Cubic_rbf(r):
    return r*r*r
        
# multiquadric radial basis function
def Multiquad_rbf(r):
    return math.sqrt(1+r*r)
    
# inverse multiquadric radial basis function
def Invmultiquad_rbf(r):
    return 1.0/math.sqrt(1+r*r)
    
# inverse quadric radial basis function
def Invquad_rbf(r):
    return 1.0/(1+r*r)
    
# Gaussian radial basis function
def Gauss_rbf(r):
    return math.exp(-r*r)

###############################################################################
# END: FUNCTION DECLARATION
###############################################################################
