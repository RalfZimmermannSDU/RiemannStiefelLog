#------------------------------------------------------------------------------
#@author: Ralf Zimmermann, IMADA, SDU Odense
# zimmermann@imada.sdu.sdk
#
# This file implements the following methods:
#
#
# * Stiefel_geodesic_interp_pre(Locs, samples, metric_alpha = 0.0):
#           - data preprocessing for geodesic interpolation
#
# * Stiefel_geodesic_interp(Locs,Deltas,samples,mu_star,metric_alpha = 0.0):
#            - compute a geodesic "polygon course" 
#             that interpolates a data set, 1D only 
#
#
#
# * Stiefel_Hermite_interp_pre(Locs, dLocs, samples, metric_alpha = 0.0, mode=1):
#           - data preprocessing for Hermite interpolation 
#
# * def Stiefel_Hermite_interp
#            - compute a composite spline of local cubic Hermite interpolants
#
#
#
# * Stiefel_RBF_tang_interp_pre(Locs, samples, metric_alpha=0.0)
#           -  data preprocessing for RBF tangent space interpolation
#
# * Stiefel_RBF_tang_interp(Locs,samples,Deltas,mu_star,Rinv,RBF_fun,metric_alpha = 0.0)
#           - map all data points to single tangent space
#              use RBF interpolation
#
#
#
# * Stiefel_Spline_interp_pre(Locs, samples, metric_alpha = 0.0):
#           - data preprocessing for 3pts spline interpolation
#
# * Stiefel_cubic_3pts_spline_interp(Locs,Deltas,Vs_bound,samples,t,metric_alpha = 0.0):
#           - evaluation of the 3pts spline course at t
#------------------------------------------------------------------------------


import numpy as np
import sys
import time

sys.path.append('../../../Stiefel_log_general_metric/SciPy/')
sys.path.append('../General_interp_tools/')

import Stiefel_Exp_Log          as StEL
import RBF_interp               as RBF
import Hermite_interp           as HI
import Spline_interp            as SI
import Stiefel_diff_tools       as Stdf

#------------------------------------------------------------------------------
# Preprocessing for geodesic interpolation:
# compute tangent space images
# 1D only
#------------------------------------------------------------------------------
def Stiefel_geodesic_interp_pre(Locs, samples, metric_alpha = 0.0):
    print('  ***   ')
    print('preprocessing of geodesic data')
    print('  ***   ')
    t_start = time.time()
    
    # numerical accuracy: To Do: should be set relative!
    tau = 1.0e-11
    
    # 0.) allocate memory
    dims = Locs.shape
    
    
    # 0.1) these are the tangent space images
    # Delta[k,:,:] = Log_{pk}(pk+1)
    Deltas = np.zeros((dims[0]-1, dims[1], dims[2]))
    
    for k in range(len(samples)-1):
        # step 1: get tangent vector via Stiefel log
        Delta, iter_conv = StEL.Stiefel_Log(Locs[k,:,:],\
                                            Locs[k+1, :,:],\
                                            tau,\
                                            metric_alpha)
        Deltas[k,:,:] = Delta
    t_end = time.time()
    print('  ***   ')
    print('Preprocessing of geodesic data finished in ', t_end-t_start, 's')
    print('  ***   ')
    return Deltas
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# compute a "polygon course" of geodesics
# 1D only
#------------------------------------------------------------------------------
def Stiefel_geodesic_interp(Locs,\
                            Deltas,\
                            samples,\
                            mu_star,\
                            metric_alpha = 0.0):
    # step 0: in which interval does mu_star belong?
    # assumption: samples is an ordered list of mu-values
    aux = abs(samples - mu_star)
    index = np.argmin(aux)
    if (mu_star < samples[index]) or abs(mu_star - samples[-1])<1.0e-15:
        pos = index-1
    else:
        pos = index

    # step 1: get tangent vector
    Delta = Deltas[pos,:,:]
    # step 2: linear interpolation in tangent space
    lin_factor = (mu_star - samples[pos])/(samples[pos+1] - samples[pos])
    # step 3: back to manifold
    U_star = StEL.Stiefel_Exp(Locs[pos,:,:], lin_factor*Delta, metric_alpha)

    return U_star
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# Preprocessing for a cubic Hermite interpolants
# * compute all the tangent space images and the shifted velocities
# 1D only
# Inputs:
#   Locs = list of sample locations
#  dLocs = list of velocities at sample locations
#samples = parameter location 
#   mode = 1:"q-centered", 0: "p-centered"
#------------------------------------------------------------------------------
def Stiefel_Hermite_interp_pre(Locs, dLocs, samples, metric_alpha = 0.0, mode=1):
    print('  ***   ')
    print('preprocessing of Hermite data')
    print('  ***   ')
    t_start = time.time()
    # 0.) allocate memory
    dims = Locs.shape
    # 0.1) these are the tangent space images
    # Delta[k,:,:] = Log_{pk+1}(pk)
    Deltas = np.zeros((dims[0]-1, dims[1], dims[2]))
    # 0.2) these are the velocities shifted to the interpolation tangent space
    Vs_shifted =  np.zeros((dims[0]-1, dims[1], dims[2]))
    
    for k in range(len(samples)-1):
        # step 1: gather sample data in interval [mu_i, mu_i+1]
        # step 1.1: get tangent vectors
        # Delta10 = tangent image of Locs[pos]
        #      0  = tangent image of Locs[pos+1]
        U0 = Locs[k,:,:]
        U1 = Locs[k+1,:,:]
    
        #get velocity vectors
        V0 = dLocs[k,:,:]
        V1 = dLocs[k+1,:,:]
        
        # default: calculations centered at q
        
        # numerical accuracy, should be set relative!
        tau = 1.0e-11
        if mode:
            Delta, iter_conv =  StEL.Stiefel_Log(U1, U0, tau, metric_alpha)
            Deltas[k,:,:] = Delta
            # step 1.2: align velocity vectors
            # in this case, V0 needs to be "moved" to T_U1 St(n,p)
            # compute V01 = d/dt|t=0   (Log_U1 o Exp_U0)(t*V0)
            
            # data for central difference approximation
            h = 1.0e-4;
            T = StEL.Stiefel_Exp(U0, h*V0, metric_alpha)
            fplus, iter_conv =  StEL.Stiefel_Log(U1, T, tau,metric_alpha)
            T2 = StEL.Stiefel_Exp(U0, -h*V0,metric_alpha)
            fminus, iter_conv =  StEL.Stiefel_Log(U1, T2, tau,metric_alpha)
            
            #  central difference approximation   
            V01 = (1.0/(2*h))*(fplus - fminus)
            Vs_shifted[k,:,:] = V01
        else: 
            # calculations centered at p
            Delta, iter_conv =  StEL.Stiefel_Log(U0, U1, tau, metric_alpha)
            Deltas[k,:,:] = Delta
            # step 1.2: align velocity vectors
            # in this case, V1 needs to be "moved" to T_U0 St(n,p)
            # compute V01 = d/dt|t=0   (Log_U0 o Exp_U1)(t*V1)
            # use central difference approximation
            h = 1.0e-4;
            T = StEL.Stiefel_Exp(U1, h*V1, metric_alpha)
            fplus, iter_conv =  StEL.Stiefel_Log(U0, T, tau, metric_alpha)
            T2 = StEL.Stiefel_Exp(U1, -h*V1)
            fminus, iter_conv =  StEL.Stiefel_Log(U0, T2, tau, metric_alpha)
            V01 = (1.0/(2*h))*(fplus - fminus)
            Vs_shifted[k,:,:] = V01
    # end of loop: for k in range(len(samples)): ...
    t_end = time.time()
    print('  ***   ')
    print('Preprocessing of Hermite data finished in', t_end-t_start, 's')
    print('  ***   ')
    return Deltas, Vs_shifted
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# compute a cubic Hermite interpolants
# between each two sample points
# 1D only
# Inputs:
#      Locs = list of sample points
#     dLocs = list of velocities at sample points
#    Deltas = tangent space images
#Vs_shifted = shifted velocities
#   samples = list of sample locations
#   mu_star = interpolation location 
#metric_alpha = metric parameter: 0.0 = canonical, -0.5 = Euclid
#      mode = 1:"q-centered", 0: "p-centered"
#------------------------------------------------------------------------------
def Stiefel_Hermite_interp(Locs,\
                           dLocs,\
                           Deltas,\
                           Vs_shifted,\
                           samples,\
                           mu_star,\
                           metric_alpha = 0.0,\
                           mode=1):
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
    # step 1.1: get tangent vectors
    # Delta10 = tangent image of Locs[pos]
    #      0  = tangent image of Locs[pos+1]
    U0 = Locs[pos,:,:]
    U1 = Locs[pos+1,:,:]
    
    #get velocity vectors
    V0 = dLocs[pos,:,:]
    V1 = dLocs[pos+1,:,:]
        
    # default: calculations centered at q
    if mode:
        # image of sample location
        Delta = Deltas[pos,:,:]
        
        # shifted velocity
        V01 = Vs_shifted[pos,:,:]
        
        # step 2: Hermite interpolation in tangent space
        # interpolant is of the form
        # H_interp(t) = a0(t)*Delta + b0(t)V01 + b1(t)V1
        H_interp = HI.Hermite_1000(mu_star, mu0, mu1)*Delta +\
                   HI.Hermite_0100(mu_star, mu0, mu1)*V01 +\
                   HI.Hermite_0001(mu_star, mu0, mu1)*V1
        # step 3: back to manifold
        U_star = StEL.Stiefel_Exp(U1, H_interp, metric_alpha)
    else: # calculations centered at p
        # image of sample location
        Delta = Deltas[pos,:,:]
        
        # shifted velocity
        V01 = Vs_shifted[pos,:,:]
    
        # step 2: Hermite interpolation in tangent space
        # interpolant is of the form
        # H_interp(t) = a0(t)*Delta + b0(t)V01 + b1(t)V1
        H_interp = HI.Hermite_0010(mu_star, mu0, mu1)*Delta +\
                   HI.Hermite_0100(mu_star, mu0, mu1)*V0 +\
                   HI.Hermite_0001(mu_star, mu0, mu1)*V01
        # step 3: back to manifold
        U_star = StEL.Stiefel_Exp(U0, H_interp, metric_alpha)

    return U_star
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# Preprocessing for RBF interpolation in tangent space:
# map all data to one tangent space for later use in interpolation procedure
# Conventions: 
#  * mu samples are listed column-wise
#    => len(samples[0]) is the number of sample points
#  * the 'mid-index' is used as tangent space center
#------------------------------------------------------------------------------
def Stiefel_RBF_tang_interp_pre(Locs, samples, metric_alpha=0.0):
    print('  ***   ')
    print('Preprocessing of tangent space interpolation data')
    print('  ***   ')
    t_start = time.time()
    # 0.) allocate memory
    Deltas = np.zeros(Locs.shape)
    # 1.) map all data to tangent space
    # use "mid point as base"
    pos0 = int(samples.shape[1]/2)
    for k in range(Deltas.shape[0]):
        # compute logarithm
        tau = 1.0e-11
        Delta, iter_conv =  StEL.Stiefel_Log(Locs[pos0,:,:],\
                                             Locs[k, :,:],\
                                             tau,\
                                             metric_alpha)  
        Deltas[k,:,:] = Delta
    # reset base point to exact zero
    Deltas[pos0,:,:] = np.zeros((Locs.shape[1], Locs.shape[2]))
    t_end = time.time()
    print('  ***   ')
    print('Preprocessing of tangent space interpolation data finished in', t_end-t_start, 's')
    print('  ***   ')
    return Deltas
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# RBF interpolation in tangent space:
# map all data to one tangent space
# perform interpolation
# Convention: mu samples are listed column-wise
# => len(samples[0]) is the number of sample points
#------------------------------------------------------------------------------
def Stiefel_RBF_tang_interp(Locs,\
                            samples,\
                            Deltas,\
                            mu_star,\
                            Rinv,\
                            RBF_fun,\
                            metric_alpha = 0.0):
    # 0.) allocate memory
    #Deltas = np.zeros(Locs.shape)
    # 1.) map all data to tangent space
    # use "mid point as base"
    pos0 = int(samples.shape[1]/2)

    # 2.) build RBF interpolator
    RBF_coeffs = RBF.RBF_mat_interp(mu_star, Rinv, samples, RBF_fun)
    
    Delta_interp = np.zeros((Locs.shape[1], Locs.shape[2]))
    for i in range(len(samples[0])):
        Delta_interp += RBF_coeffs[i]*Deltas[i,:,:]
            

    # step 3: back to manifold
    U_star = StEL.Stiefel_Exp(Locs[pos0,:,:], Delta_interp, metric_alpha)

    return U_star
#------------------------------------------------------------------------------   




#------------------------------------------------------------------------------
# Preprocessing for a cubic-spline interpolants
# * compute all the tangent space images and the shifted velocities
# 1D only
# Inputs:
#         Locs = list of sample locations
#      samples = parameter location 
# metric_alpha = metric parameter
#
# Ouputs:
#       Deltas = tangent space images
#      Vbounds = velocity vectors at the connectors of the spline interpolant 
#------------------------------------------------------------------------------
def Stiefel_Spline_interp_pre(Locs, samples, metric_alpha = 0.0):
    print('  ***   ')
    print('preprocessing of cubic spline data')
    print('  ***   ')
    
    #numerical accuracy for Stiefel log operations
    tau = 1.0e-11
    
    t_start = time.time()
    # 0.) allocate memory
    dims = Locs.shape
    
    
    print("Dimensions", dims )
    
    if (dims[0] % 2) == 0:
        print("Warning: odd number of sample points needed!")
    # 0.1) these are the tangent space images
    #      map p0,p1,p2 to T_p1,    p1 is mapped to zero
    #      map p2,p3,p4 to T_p3,    p3 is mapped to zero
    #      map p4,p5,p6 to T_p5,    p3 is mapped to zero, etc.
    #      do not store the zeros
    #      storing scheme:
    #       Deltas[0,1] : [p0,p2]   on T_p1
    #       Deltas[2,3] : [p2,p4]   on T_p2
    #       Deltas[4,5] : [p4,p6]   on T_p3  etc.
    #
    #      for l samples, l >= 3, need 2*(l-1)/2 tangent vectors
    Deltas = np.zeros((dims[0]-1, dims[1], dims[2]))
    # 0.2) these are the boundary velocities shifted 
    #      compute dp2 in the interpolation tangent space
    #      storage scheme:
    #      dp2 in T_p3 stored in Vs_bound[0]
    #      dp4 in T_P5 stored in Vs_bound[1]
    #      dp6 in T_P7 stored in Vs_bound[2] etc.
    #
    #      for l samples, need (l-3)/2 tangent vectors
    Vs_bound =  np.zeros((int((dims[0]-3)/2), dims[1], dims[2]))
    
    # initial triplet
    k = 0
    U0      = Locs[k,:,:]
    Ucenter = Locs[k+1,:,:]
    U2      = Locs[k+2,:,:]
    
    # map to tangent space at Ucenter
    Deltas[k,:,:], iter_conv   = StEL.Stiefel_Log(Ucenter, U0,\
                                                  tau, metric_alpha)
    Deltas[k+1,:,:], iter_conv = StEL.Stiefel_Log(Ucenter, U2,\
                                                  tau, metric_alpha)
    
    print("Delta",k, "is tangent image of Locs",k," in T_U", k+1)
    print("Delta",k+1, "is tangent image of Locs",k+2," in T_U", k+1)
    # compute boundary derivative at sample t2:
    da0,da2,db0    = SI.Cubic_spline_diff_coeffs(k, samples, 0)
    # this gives a tangent vector in T_p1:
    V = da0 * Deltas[k,:,:] + da2 * Deltas[k+1,:,:]
    # ***
    # now, compute derivative of the manifold curve
    # c(t) = Exp_{p1}(s1(t)) = Exp_{P1}( s1(t2) + (t-t1)*s1'(t2) + O( (t-t2)^2))
    # Recall: s1(t2) = Deltas[1,:,:]
    # ***
    dSt_Exp = Stdf.Stiefel_diff_exp(Ucenter, Deltas[k+1,:,:], V, metric_alpha)
    
        
    # need to move this from T_p2 to a suitable tangent vector in T_p3
    # compute Vp3 = d/dt|t=0   (Log_p3 o Exp_U2)(t* dSt_Exp)
    # use central difference approximation
    h = 1.0e-4;
    T                = StEL.Stiefel_Exp(U2, h*dSt_Exp, metric_alpha)
    fplus, iter_conv = StEL.Stiefel_Log(Locs[3], T, tau, metric_alpha)
    T2               = StEL.Stiefel_Exp(U2, -h*dSt_Exp, metric_alpha)
    fminus,iter_conv = StEL.Stiefel_Log(Locs[3],T2, tau, metric_alpha)
    Vs_bound[k,:,:] = (1.0/(2*h))*(fplus - fminus)  
    
    # skip ahead in steps of 2
    k = 2
    while k < (dims[0]-3):
        # step 1: gather sample data for triplet tk, tk+1, tk+2
        # step 1.1: get tangent vectors
        U0      = Locs[k,:,:]
        Ucenter = Locs[k+1,:,:]
        U2      = Locs[k+2,:,:]
        
        # map to tangent space at Ucenter
        Deltas[k,:,:],  iter_conv =  StEL.Stiefel_Log(Ucenter, U0,\
                                                      tau, metric_alpha)
        Deltas[k+1,:,:],iter_conv =  StEL.Stiefel_Log(Ucenter, U2,\
                                                      tau, metric_alpha)
        
        print("Delta",k, "is tangent image of Locs",k," in T_U", k+1)
        print("Delta",k+1, "is tangent image of Locs",k+2," in T_U", k+1)
        
        # compute boundary derivative at sample tk+2:
        da0,da2,db0    = SI.Cubic_spline_diff_coeffs(k, samples, 1)
        
        # this gives a tangent vector in T_pk+1:
        V = da0 * Deltas[k,:,:] + da2 * Deltas[k+1,:,:] + db0*Vs_bound[int(k/2)-1,:,:]
        # ***
        # now, compute derivative of the manifold curve
        # c(t) = Exp_{p_{k+1}}(s_{k+1}(t)) 
        #      = Exp_{P_{k+1}}(s_{k+1}(t_{k+2}) + (t-t_{k+2})*s_{k+1}'(t_{k+2}) + O( (t-t_{k+2})^2))
        # Recall: s_{k+1}(t_{k+2}) = Deltas[k+1,:,:]
        # ***
        dSt_Exp = Stdf.Stiefel_diff_exp(Ucenter, Deltas[k+1,:,:], V,\
                                        metric_alpha)
    
        
        # need to move this to a suitable tangent vector in T_p{k+3}
        # compute Vp{k+3} = d/dt|t=0   (Log_p{k+3} o Exp_Ucenter)(t* dSt_Exp)
        # use central difference approximation
        h = 1.0e-4;
        T                = StEL.Stiefel_Exp(U2, h*dSt_Exp, metric_alpha)
        fplus, iter_conv = StEL.Stiefel_Log(Locs[k+3], T, tau, metric_alpha)
        T2               = StEL.Stiefel_Exp(U2, -h*dSt_Exp, metric_alpha)
        fminus,iter_conv = StEL.Stiefel_Log(Locs[k+3],T2, tau, metric_alpha)
        Vs_bound[int(k/2),:,:] = (1.0/(2*h))*(fplus - fminus) 
 
        #increase counter in steps of 2
        k = k+2

    
    #final samples do not need a forward of derivatives
    # map to tangent space at Ucenter
    Deltas[k,:,:],  iter_conv = StEL.Stiefel_Log(Locs[k+1,:,:], Locs[k,:,:],\
                                                 tau, metric_alpha)
    Deltas[k+1,:,:],iter_conv = StEL.Stiefel_Log(Locs[k+1,:,:], Locs[k+2,:,:],\
                                                 tau, metric_alpha)
        
    print("Delta",k, "is tangent image of Locs",k," in T_U", k+1)
    print("Delta",k+1, "is tangent image of Locs",k+2," in T_U", k+1)
    
    # end of loop: for k in range(len(samples)): ...
    t_end = time.time()
    print('  ***   ')
    print('Preprocessing of spline data finished in', t_end-t_start, 's')
    print('  ***   ')
    
    
    for k in range(Vs_bound.shape[0]):
        print(" ")
    return Deltas, Vs_bound
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# compute a cubic 3pts spline course
# 
# 1D only
# Inputs:
#      Locs = list of sample points
#    Deltas = tangent space images
#             Deltas[0,1] : [p0,p2]   on T_p1
#             Deltas[2,3] : [p2,p4]   on T_p2
#             Deltas[4,5] : [p4,p6]   on T_p3  etc.
#  Vs_bound = velocities at left boundary of triple tk, tk+1, tk+2
#             dp2 in T_p3 stored in Vs_bound[0]
#             dp4 in T_P5 stored in Vs_bound[1]
#             dp6 in T_P7 stored in Vs_bound[2] etc.
#   samples = list of sample locations
#   mu_star = interpolation location 
#metric_alpha = metric parameter: 0.0 = canonical, -0.5 = Euclid
#
#------------------------------------------------------------------------------
def Stiefel_cubic_3pts_spline_interp(Locs,\
                                     Deltas,\
                                     Vs_bound,\
                                     samples,\
                                     t,\
                                     metric_alpha = 0.0):
    # step 0: in which interval does mu_star belong?
    # assumption: samples is an ordered list of mu-values
    # find position in interval
    # !! number of samples must be odd: 2k+1 with k>=1 !!
    aux = t - samples
    # t is in the interval, where the first entry of "aux" is negative
    index = len(aux[aux>0])-1
    index = 2*int(index/2)
    
    # local sample triplet    
    t0 = samples[index]
    t1 = samples[index+1]
    t2 = samples[index+2]
    
    if 0:
        print("for t = ", t, "use", "(t0,t1,t2)=", index, t0,\
                                                   index+1, t1,\
                                                   index+2, t2)
        
    # evaluate the tangent space spline at t
    if index == 0:
        # start interval: no derivatives
        a0,a2,b0 = SI.Cubic_spline_coeffs(t, np.array([t0,t1,t2]), 0)  
        tang_spline_t = a0*Deltas[index,:,:] + a2*Deltas[index+1,:,:]
    else:
        a0,a2,b0 = SI.Cubic_spline_coeffs(t, np.array([t0,t1,t2]), 1)
                
        if 0:
            print("index:", index, "int(index/2):", int(index/2))
            print("Coeffs:",a0,a2,b0)
        

            #test
            T  = np.dot(Locs[index+1].T, Deltas[index,:,:])
            nD1= np.linalg.norm(T+T.T, 'fro')
            T  = np.dot(Locs[index+1].T, Deltas[index+1,:,:])
            nD2= np.linalg.norm(T+T.T, 'fro')
            T  = np.dot(Locs[index+1].T, Vs_bound[int(index/2)-1,:,:])
            nV = np.linalg.norm(T+T.T, 'fro')
            print("all in same tang space?", index, ": Is in proper tang space?", nD1, nD2, nV)
        
            print("Used Vs_bound[",int(index/2)-1,"] for t = ", t)
            #end: test   
    
        tang_spline_t = a0*Deltas[index,:,:] + a2*Deltas[index+1,:,:] + b0*Vs_bound[int(index/2)-1,:,:]
         
    # step 3: map back to manifold
    U_star = StEL.Stiefel_Exp(Locs[index+1], tang_spline_t, metric_alpha)

    return U_star
#------------------------------------------------------------------------------