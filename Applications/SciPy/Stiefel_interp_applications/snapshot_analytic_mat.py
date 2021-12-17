#
# This file implements some analytic matrix test function
# that feature in the numerical experiments of
# 
# HERMITE INTERPOLATION AND DATA PROCESSING ERRORS ON RIEMANNIAN MATRIX MANIFOLDS
# SIAM J. SCI . COMPUT. Vol. 42, No. 5, pp. A2593--A2619
#
# @author: R. Zimmermann
import numpy as np
import math
import scipy
from   scipy import linalg
import sys

sys.path.append('../../../Stiefel_log_general_metric/SciPy/')
import Stiefel_diff_tools as sdiffs






def snapshot_analytic_mat2(t_range, mu, dim, RefU, comp_deriv, key):
    #
    # this function produces mu-dependent snapshot matrices of dimension 
    #  dim x len(t_range)
    # together with their derivatives and 
    # QR- or SV-decompositions, respectively.
    # 
    # inputs:
    #  t_range = discrete parameter interval
    #  mu      = operating condition parameter
    #  dim     = matrix dimension
    #  RefU    = reference U (or Q) matrix to align the signs, 
    #            possibly "None"
    #  comp_deriv = boolean: compute derivative matrices 0/1
    #  key     = string, either 'SVD' or 'QR'
    #
    # Outputs
    # U, dU, where U stems either from SVD or QR 
    #
    x = np.linspace(0.0,1.0,dim)
    Y = np.zeros( (dim, len(t_range))) 
    dY= np.zeros( (dim, len(t_range)))  
    # compute snapshot matrices
    for k in range(len(t_range)):
        #
        #
        t = t_range[k]
        # create f-data
        # all operations work elementwise in Python
        y = (x**t)*np.sin((np.pi/2.0)*mu*x)

        dx = x[1]-x[0]
        norm_y_L2 = (math.sqrt(dx)*linalg.norm(y[1:-1],2))
        Y[:,k] =y/norm_y_L2

        if comp_deriv:        
            #compute derivative by mu
            dydmu = (np.pi/2.0)*(x**(t+1.0))*np.cos((np.pi/2.0)*mu*x)
        
            inner = dx*np.dot(y[1:-1].T, dydmu[1:-1])
        
            dY[:,k] = (1.0/norm_y_L2)*dydmu - \
                      (inner/norm_y_L2**3)*y
                  
    if key == 'SVD':        
        # compute SVD of snapshot matrix
        U, Sigma, VT = scipy.linalg.svd(Y,\
                                       full_matrices=False,\
                                       compute_uv=True,\
                                       overwrite_a=True)
        if RefU[0,0] != None:
            #try and align the coordinates
            Coord = np.dot(U.T, RefU)
            Csign = np.diag(np.sign(np.diag(Coord)))
        
            U  = np.dot(U,Csign)
            VT = np.dot(Csign, VT)
        
        if comp_deriv:
            #compute derivative data
            dU_bot, A, dU, Alpha, dSigma, dV = sdiffs.Stiefel_diff_SVD(Y, dY, U, Sigma, VT.T)
            return U, dU, Sigma,dSigma, dV, VT.T
        else:
            return U, Sigma, VT.T
            
        
    elif key == 'QR':
        if comp_deriv:
            # QR decomp and derivative of QR decomp
            U, dU, R, dR = sdiffs.Stiefel_diff_QR(Y, dY)
    
            #T = np.dot(U.T, dU)
            #print("dU on tangent space?", linalg.norm(T+T.T, 'fro'))
            return U, dU
        else:
            # qr of base matrix
            U, R = linalg.qr(Y, overwrite_a=True,\
                             lwork=None,\
                             mode='economic',\
                             pivoting=False,\
                             check_finite=True)
            return U
    else:
        print('Key for matrix decomposition not defined. Choose either QR or SVD')
        return 0






def snapshot_analytic_mat3(t, Y0, Y1, Y2, Y3, comp_deriv, key):
    #
    # this function produces a t-dependent matrix curve
    #  t |-> Y_0 + t Y_1 + t^2Y_2 + t^3 Y_3 
    # where all the matrices Y_i are nxp
    
    Yt = Y0 + t*Y1 + (t**2)*Y2 + (t**3)*Y3
    #Yt = Yt/np.linalg.norm(Yt, 'fro')
    if comp_deriv:
        dYt = Y1 + (2*t)*Y2 + (3*t**2)*Y3   

    if key == 'SVD':
        #SVD of Y(0)
        U0, Sigma0, VT0 = scipy.linalg.svd(Y0,\
                                       full_matrices=False,\
                                       compute_uv=True,\
                                       overwrite_a=True)
        
        
        # compute SVD of snapshot matrix at t
        U, Sigma, VT = scipy.linalg.svd(Yt,\
                                       full_matrices=False,\
                                       compute_uv=True,\
                                       overwrite_a=True)
        #try and align the coordinates
        Coord = np.dot(U.T, U0)
        Csign = np.diag(np.sign(np.diag(Coord)))
        
        U  = np.dot(U,Csign)
        VT = np.dot(Csign, VT)
        #----------------------------------

        
        if comp_deriv:
            #compute derivative data
            dU_bot, A, dU, Alpha, dSigma, dV = sdiffs.Stiefel_diff_SVD(Yt, dYt, U, Sigma, VT.T)
            return U, dU, Sigma, dSigma, VT.T, dV
        else:
            return U, Sigma, VT.T
        
    elif key == 'QR':
        if comp_deriv:
            # QR decomp and derivative of QR decomp
            U, dU, R, dR = sdiffs.Stiefel_diff_QR(Yt, dYt)
    
            T = np.dot(U.T, dU)
            print("dU on tangent space?", linalg.norm(T+T.T, 'fro'))
            return U, dU
        else:
            # qr of base matrix
            U, R = linalg.qr(Yt, overwrite_a=True,\
                               lwork=None,\
                               mode='economic',\
                               pivoting=False,\
                               check_finite=True)
            return U
    else:
        print('Key for matrix decomposition not defined. Choose either QR or SVD')
        return 0










def snapshot_analytic_lowrank(t, Y0, Y1, Y2, Y3, Z0, Z1, Z2, comp_deriv, key):
    #
    # this function produces a t-dependent lowrank matrix curve
    #  t |-> (Y_0 + t Y_1 + t^2Y_2 + t^3 Y_3)*(Z0 + tZ1)
    # where all the matrices Y_i are nxp and Z_i are (pxm)
    
    Yt = Y0 + t*Y1 + (t**2)*Y2 + (t**3)*Y3
    dYt = Y1 + (2*t)*Y2 + (3*t**2)*Y3
    Zt = Z0+t*Z1 + (t**2)*Z2
    dZt = Z1 + (2*t)*Z2
    
    Wt = np.dot(Yt, Zt)
    W0 = np.dot(Y0, Z0)
    if comp_deriv:
        dWt = np.dot(dYt, Zt) + np.dot(Yt, dZt)

    if key == 'SVD':
        #SVD of Y(0)
        U0, Sigma0, VT0 = scipy.linalg.svd(W0,\
                                           full_matrices=False,\
                                           compute_uv=True,\
                                           overwrite_a=True)
        
        
        # compute SVD of snapshot matrix at t
        U, Sigma, VT = scipy.linalg.svd(Wt,\
                                        full_matrices=False,\
                                        compute_uv=True,\
                                        overwrite_a=True)

        #try and align the coordinates
        Coord = np.dot(U.T, U0)
        Csign = np.diag(np.sign(np.diag(Coord)))
        U = np.dot(U,Csign)
        VT = np.dot(Csign, VT)
        
        # TRUNCATE THE SVD
        tau = 1.0e-5
        St = Sigma[Sigma>tau]
        p = St.shape[0]
        # keep only the first p columns
        Ut = np.matrix(U[:,0:p])
        Vt = np.matrix(VT.T[:,0:p])
        
        
        if comp_deriv:
            #compute derivative data
            #diff-algorithm needs full orthogonal V
            dU_bot, A, dU, Alpha, dSigma, dV = sdiffs.Stiefel_diff_SVD(Wt, dWt, Ut, St, VT.T)
            
            return Ut, dU, St, dSigma, Vt, dV
        else:
            return Ut, St, Vt
        
    elif key == 'QR':
        if comp_deriv:
            # QR decomp and derivative of QR decomp
            U, dU, R, dR = sdiffs.Stiefel_diff_QR(Wt, dWt)
            return U, dU
        else:
            # qr of base matrix
            U, R = linalg.qr(Wt, overwrite_a=True,\
                               lwork=None,\
                               mode='economic',\
                               pivoting=False,\
                               check_finite=True)
            return U
    else:
        print('Key for matrix decomposition not defined. Choose either QR or SVD')
        return 0





