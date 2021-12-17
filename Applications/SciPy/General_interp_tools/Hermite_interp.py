
import numpy as np
import sys

#---------------------------------------------------------------------------
#
# This file implements the coefficient functions for performing
# Hermitian interpolation between   TWO POINTS   p, q=0 (vectors/matrices) 
# in a vector space with prescribed derivatives vp, vq.
#
# This is used for interpolation in a tangent space,
# where one of the data points corresponds to the origin 0 !
#
# Approach:
# gamma(t) = a0(t)*p + b0(t)*vp + b1(t)*vq
# 
# with gamma(t0) = p, gamma(t1) = 0
#
# requirements:
#
#  f: t0     t0     t1     t1
#  a0 1      0       0      0    => Hermite_1000
#  a1 0      0       1      0    => Hermite_0010
#  b0 0      1       0      0    => Hermite_0100
#  b1 0      0       0      1    => Hermite_0001
#  f(t0)  f'(t0)  f(t1)  f'(t1)
#---------------------------------------------------------------------------


###############################################################################
#  coefficient functions for Hermite interpolation on 2 points
###############################################################################


#------------------------------------------------------------------------------
# compute Hermite polynomial of coefficient function a0
# with
#     a0  a0' 
# t0: 1   0
# t1: 0   0
#------------------------------------------------------------------------------
def Hermite_1000(t, t0, t1):
    a0 =  1.0-(1.0/(t1-t0)**2)*(t-t0)**2 +\
          (2.0/(t1-t0)**3)*(t-t0)**2*(t-t1)
    return a0
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# compute Hermite polynomial of coefficient function a0
# with
#     a0  a0' 
# t0: 0   0
# t1: 1   0
#------------------------------------------------------------------------------
def Hermite_0010(t, t0, t1):
    a1 =  (1.0/(t1-t0)**2)*(t-t0)**2 -\
          (2.0/(t1-t0)**3)*(t-t0)**2*(t-t1)
    return a1
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# compute Hermite polynomial of coefficient function b0
# with
#     b0  b0' 
# t0: 0   1
# t1: 0   0
#------------------------------------------------------------------------------
def Hermite_0100(t, t0, t1):
    b0 =  (t-t0) - (1.0/(t1-t0))*(t-t0)**2 + \
          (1.0/(t1-t0)**2)*((t-t0)**2)*(t-t1)
    return b0
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# compute Hermite polynomial of coefficient function b1
# with
#     b1  b1' 
# t0: 0   0
# t1: 0   1
#------------------------------------------------------------------------------
def Hermite_0001(t, t0, t1):
    b1 =  (1.0/(t1-t0)**2)*(t-t0)**2*(t-t1)
    return b1
#------------------------------------------------------------------------------


###############################################################################
# coefficient functions for Hermite interpolation on 2 points
###############################################################################



###############################################################################
# FUNCTIONs for general (combined Lagrange-) Hermite Interpolation
###############################################################################


#------------------------------------------------------------------------------
#
# compute the coefficients in Newton's representation
# of the unique interpolation polynomial via the divided differences
#
# input arguments:
#   sample_x : sample locations
#              (multiple locations indicate the use of derivatives)
#   sample_f : corresponding sample values
# output arguments:
#         dd : coefficients w.r.t. Newton's basis
#------------------------------------------------------------------------------

def NewtonCoeff(sample_x, sample_f, sample_df):
    # get dimension
    n=len(sample_x)
    # check if there are as many sample locs as there are sample values
    if abs(len(sample_f) - n) > 0:
        print('Two vectors of same length are needed as input!')
        sys.exit()

    #-------------------------------------------
    # compute coeffs via divided differences dd
    # CONVENTION 
    # The data arrives as in this example
    #  t0,  t0, t1, t2, t3, t3
    #  f0, f0', f1, f2, f3, f3'
    # no higher order derivatives
    #-------------------------------------------

    # first column in dd scheme (k=1):
    # copy the sample values to the dd-vector
    dd=np.array(sample_f)
    
    # next is column 2
    # however, we need only the first entry from column 1
    # and only the entries 2,...,n from column 2
    #   => we store entry one and overwrite the other ones
    #      this process is then repeated.   
    for l in range(1,n):
        # proceed backwards
        j =n-1
        while j >= l:
            # only first order derivatives expected!!!
            if (sample_x[j] == sample_x[j-1]) and (l==1):
                # use derivative data
                dd[j] = sample_df[sample_x[j]]
            else:
                dd[j] = (dd[j]-dd[j-1])/(sample_x[j]-sample_x[j-l])
            j = j-1   
    return dd
# END -------------------------------------------------------------------------


#------------------------------------------------------------------------------
# evaluate the Newton polynomial p at xstar
#   via nested multiplication (Horner's scheme)
#
# input arguments:
# divdiff  : divided difference coefficients
# sample_x : vector of sample locations
#  xstar   : location where the poly is evaluated
# output arguments:
#   result : the function value p(x)
#------------------------------------------------------------------------------
def NewtonInterp(divdiff, sample_x, xstar):

    # start with the last entry
    result=divdiff[-1];

    # sum up the terms backwards
    n = len(divdiff)
    i = n-2
    while i >= 0:
        result=result*(xstar-sample_x[i])+divdiff[i]
        i = i-1
    return result
#END --------------------------------------------------------------------------


#------------------------------------------------------------------------------
# compute cubic Hermite interpolants
# between each two sample points
# 1D only
# Inputs:
#   Locs = list of sample locations, here sample vectors
#  dLocs = list of velocity vectors at sample locations
# mu_star= interpolation location 
#------------------------------------------------------------------------------
def Hermite_vec_interp(Locs, dLocs, samples, mu_star):
    # step 0: in which interval does mu_star belong?
    # assumption: samples is an ordered list of mu-values
    aux = abs(samples - mu_star)
    index = np.argmin(aux)
    if (mu_star < samples[index]) or abs(mu_star - samples[-1])<1.0e-15:
        pos = index-1
    else:
        pos = index
    # interval bounds
    mu0 = samples[pos]
    mu1 = samples[pos+1]
    # step 1: gather sample data in interval [mu_i, mu_i+1]
    if len(Locs.shape) == 2:
        # vector data
        v0  =  Locs[pos,:]
        v1  =  Locs[pos+1,:]
        dv0 = dLocs[pos,:]
        dv1 = dLocs[pos+1,:]
    elif len(Locs.shape) == 3:
        #matrix data
        v0  =  Locs[pos,:,:]
        v1  =  Locs[pos+1,:,:]
        dv0 = dLocs[pos,:,:]
        dv1 = dLocs[pos+1,:,:]
    else:
        print('Data format not supported by Hermite_vec_interp')

    # step 2: Hermite interpolation 
    # interpolant is of the form
    # H_interp(t) = a0(t)*v0 + + a1(t)*v1 + b0(t)dv0 + b1(t)dv1 
    H_interp = Hermite_1000(mu_star, mu0, mu1)*v0  +\
               Hermite_0010(mu_star, mu0, mu1)*v1  +\
               Hermite_0100(mu_star, mu0, mu1)*dv0 +\
               Hermite_0001(mu_star, mu0, mu1)*dv1

    return H_interp
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# compute n chebychev roots in the interval [a,b]
#------------------------------------------------------------------------------
def ChebyRoots(a, b, n):
    chebyroots = np.zeros((n,))
    
    # in [-1,1]-interval
    for i in range(1,n+1):
        # filling the array from the rear gives the roots in ascending order
        chebyroots[(n-1)- (i-1)] = np.cos((2*i-1)*np.pi/(2*n))
    # translate to [a,b]
    chebyroots = 0.5*(b-a)*chebyroots + 0.5*(b+a)
    return chebyroots
#END --------------------------------------------------------------------------
