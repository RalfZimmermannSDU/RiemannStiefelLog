import numpy as np


#------------------------------------------------------------------------------
#@author: Ralf Zimmermann, IMADA, SDU Odense
# zimmermann@imada.sdu.sdk
#
# non-Riemannian tools for interpolation on a vector space
#
#
#
# This file implements the following methods:
#
# * matrix_linear_interp(Locs, samples, mu_star):
#       - linear interpolation of vector- (matrix)-valued data points
#
# * Cubic_spline_coeffs(t, samples, diff_samples = None):
#       - create the coeficient functions for a cubic spline on three points
#
# * Cubic_spline_diff_coeffs(pos, samples, use_diff):
#       - derivatives of spline coefficients
#
# * Cubic_spline_3pts_course(t, samples, Locs, diff_samples):
#       - computes an interpolant via a connected course of local three-points
#         splines
#------------------------------------------------------------------------------



###############################################################################
# FUNCTION DECLARATION
###############################################################################


#------------------------------------------------------------------------------
# perform linear interpolation for vector-or matrix valued data
#------------------------------------------------------------------------------
def matrix_linear_interp(Locs, samples, mu_star):
    # step 0: in which interval does mu_star belong?
    # assumption: samples is an ordered list of mu-values
    aux = abs(samples - mu_star)
    index = np.argmin(aux)
    if (mu_star < samples[index]) or abs(mu_star - samples[-1])<1.0e-15:
        pos = index-1
    else:
        pos = index

    # step 2: linear interpolation 
    lin_factor = (mu_star - samples[pos])/(samples[pos+1] - samples[pos])
    
    if len(Locs.shape) == 3:
        # list of matrices
        Loc_star = (1-lin_factor)*Locs[pos,:,:] + lin_factor * Locs[pos+1,:,:]
    elif len(Locs.shape) == 2:
        # list of vectors
        Loc_star = (1-lin_factor)*Locs[pos,:] + lin_factor * Locs[pos+1,:]
    else:
        print('Error in <matrix_linear_interp>: data points must be matrices or vectors.')

    return Loc_star
#------------------------------------------------------------------------------




#---------------------------------------------------------------------------
#
# This file implements the coefficient functions for performing
# cubic spline interpolation between three points f0, f1, f2 (vectors/matrices) 
# in a vector space with prescribed boundary conditions
#
# 
# This is used for interpolation in a tangent space,
# where the center data point corresponds to the origin f1=0 !
#
# Approach: piecewise splines
# s0(t) = A0 + (t-t0) B0 + (t-t0)^2 C0 + (t-t0)^3 D0
# s1(t) =  0 + (t-t0) B1 + (t-t0)^2 C1 + (t-t0)^3 D1
# Defining properties
#     s0(t0)  = f0 = A0
#     s0(t1)  = f1
#     
##    s1(t1)  = 0 = A1
#     s1(t2)  = f2
#
# Boundary conditions:
#
# natural: 
#   s0'' = 0 = s2''  
# (This determines s1'' uniquely via 
#   2(h0+h1)s1'' = g1  = 6/h1 f2 + 6/h0 f0
#
# semi-complete: requires derivative f0' at sample f0
#                     s2'' = 0 
#   h0 s0'' + 2(h0+h1)s1'' = g1
# 2h0s0'' + h0 s1''        = g0 = -6f0' - 6/h0 f0 
#
#
#---------------------------------------------------------------------------
def Cubic_spline_coeffs(t, samples, use_diff = 0):
    #
    # create coeffs for
    # s(t) = a0(t) f0 + a2(t) f2 ( + b0(t) f0')
    # 
    #
    t0 = samples[0]
    t1 = samples[1]
    t2 = samples[2] 
    h0 = t1-t0
    h1 = t2-t1
    #
    #
    if use_diff == 0:
        #
        inv_2h0ph1 = 1.0/(2*(h0 + h1))
        # t1 is the gluing point
        if (t>= t0) and (t < t1):
            # coeffs of s0(t)
            a0 = 1.0 -\
                 ( inv_2h0ph1 + (1.0/h0))  * (t-t0) +\
                 (1.0/(h0*h0))* inv_2h0ph1 * (t-t0)**3
                 #     
            a2 = (-h0/h1)     * inv_2h0ph1 * (t-t0) +\
                 (1.0/(h0*h1))* inv_2h0ph1 * (t-t0)**3
        elif (t<=t2):
            #coeffs of s1(t)
            a0 = (-2.0*h1/h0)  * inv_2h0ph1 * (t-t1)    +\
                 (3.0/h0)      * inv_2h0ph1 * (t-t1)**2 +\
                 (-1.0/(h0*h1))* inv_2h0ph1 * (t-t1)**3
                 #
            a2 =  (2.0*h0/h1)  * inv_2h0ph1 * (t-t1)    +\
                 (3.0/h1)      * inv_2h0ph1 * (t-t1)**2 +\
                 (-1.0/(h1*h1))* inv_2h0ph1 * (t-t1)**3
        else:
            print("Wrong t input in <Cubic_spline_coeffs>")
            a0 = 0.0
            a2 = 0.0
        b0 = 0.0
    else:
        inv_3h0p4h1 = 1.0/(3*h0 + 4*h1)
        if (t>= t0) and (t <= t1):
            a0 = 1.0 \
                 - (3.0/(2*h0*h0) + (9.0/(2*h0))*inv_3h0p4h1)       * (t-t0)**2 \
                 + (1.0/(2*h0*h0*h0) + (9.0/(2*h0*h0))*inv_3h0p4h1) * (t-t0)**3
                 #     
            a2 = (-3.0/h1) *inv_3h0p4h1  * (t-t0)**2\
                 + (3.0/(h0*h1))*inv_3h0p4h1 * (t-t0)**3
            #
            b0 = (t-t0)\
                 - (3.0/(2*h0) + (3.0/2.0)*inv_3h0p4h1)        * (t-t0)**2\
                 + (1.0/(2*h0*h0) + (3.0/(2*h0))*inv_3h0p4h1)  * (t-t0)**3
        elif (t<=t2):
            #
            a0 = (-6.0*h1/h0)   * inv_3h0p4h1 * (t-t1)\
                + (9.0/h0)      * inv_3h0p4h1 * (t-t1)**2\
                + (-3.0/(h1*h0))* inv_3h0p4h1 * (t-t1)**3
                #
            a2 =  (3.0*h0/h1)   * inv_3h0p4h1 * (t-t1)\
                + (6.0/h1)      * inv_3h0p4h1 * (t-t1)**2\
                + (-2.0/(h1*h1))* inv_3h0p4h1 * (t-t1)**3
                #
            b0 = (-2.0*h1)     * inv_3h0p4h1  * (t-t1)\
                + 3.0          * inv_3h0p4h1  * (t-t1)**2\
                +(-1.0/h1)     * inv_3h0p4h1  * (t-t1)**3
        else:
            print("Wrong t input in <Cubic_spline_coeffs>")
            a0 = 0.0
            a2 = 0.0
            b0 = 0.0
    return a0, a2, b0
#------------------------------------------------------------------------------



def Cubic_spline_diff_coeffs(pos, samples, use_diff):
    # compute the derivaties of the coeffs.
    # the first one (pos =0) is determined by natural boundary conditions
    #
    # pos must be the valid start index of a sample triple
    # pos = 0,2,4,...,len(samples)-2
    #
    # the input "pos" produces the coeff-derivatives at sample pos+2 
    # as an output
    #
    if use_diff == 0:
        print("compute right-boundary derivative for start triplet",\
              pos, pos+1, pos+2)
        # compute s1'(t2)
        t0 = samples[pos]
        t1 = samples[pos+1]
        t2 = samples[pos+2] 
        h0 = t1-t0
        h1 = t2-t1
        inv_2h0ph1 = 1.0/(2*(h0 + h1))
        #
        # it holds: s1(t) = a0(t) f0 + a2(t) f2
        #   =>    s1'(t2) = a0'(t2) f0 + a2'(t2) f2
        # derivatives of coefficients:
        da0 = h1/h0   * inv_2h0ph1
        #
        da2 = (2.0*h0/h1)  * inv_2h0ph1  +\
               3.0         * inv_2h0ph1
        # dummy value
        db0 = 0.0
    else:
        print("compute right-boundary derivative for triplet",\
              pos, pos+1, pos+2)
        t0 = samples[pos]
        t1 = samples[pos+1]
        t2 = samples[pos+2] 
        h0 = t1-t0
        h1 = t2-t1
        inv_3h0p4h1 = 1.0/(3*h0 + 4*h1)
        da0 = (3.0*h1/h0)      * inv_3h0p4h1
        #
        da2 = ((3*h0/h1)+ 6.0)* inv_3h0p4h1 
        #
        db0 = h1 * inv_3h0p4h1
    return da0, da2, db0
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
def Cubic_spline_3pts_course(t, samples, Locs, diff_samples):
    # works for scalar data or vector-space data  (vectors/matrices)
    #
    # find position in interval
    # !! number of samples must be ODD !!: 2k+1 with k>=1 !!
    aux = t - samples
    # t is in the interval, where the first entry of "aux" is negative
    index = len(aux[aux>0])-1
    index = 2*int(index/2)
    
    # local sample triplet    
    t0 = samples[index]
    t1 = samples[index+1]
    t2 = samples[index+2]
    #print("for t = ", t, "use", "(t0,t1,t2)=", t0,t1,t2)
    
    if index == 0:
        # start interval: no derivatives
        a0,a2,b0 = Cubic_spline_coeffs(t, np.array([t0,t1,t2]), 0)
    
        spline_t = a0*Locs[index] + a2*Locs[index+2]
    else:
        a0,a2,b0 = Cubic_spline_coeffs(t, np.array([t0,t1,t2]), 1)
    
        spline_t = a0*Locs[index] + a2*Locs[index+2] + b0*diff_samples[index]
    return spline_t
#------------------------------------------------------------------------------