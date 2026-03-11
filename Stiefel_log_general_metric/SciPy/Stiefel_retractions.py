#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 15:37:51 2026

@author: zimmermann
"""


import numpy as np
import scipy
from   scipy import linalg
import Stiefel_Aux        as StAux


#------------------------------------------------------------------------------
# Polar factor retraction
# PF_U0 (Xi ) = (U0 + Xi)(In + XiT Xi )^-1/2
# 
# Input arguments      
#          U0    : base point on St(n,p)
#          Xi : tangent vector in T_U0 St(n,p)
# Output arguments
#          U1    : PF_U0(Xi)
#------------------------------------------------------------------------------
def Stiefel_PF_ret(U0, Xi):
#------------------------------------------------------------------------------
    # get dimensions
    n,p = U0.shape
    
    # QR decomposition of Xi,
    # only R is needed
    S = np.linalg.qr(Xi, mode='r')
    
    fast = 1
    if fast:
        # symmetric EVD 
        STS = np.dot(S.T, S)  
        Lambda, V  = scipy.linalg.eigh(STS)
        # sqrt( 1/(1+lambda)), addition is elementwise
        Lambda = np.sqrt(1/(Lambda + 1))
        STS = np.dot(V*Lambda, V.T)
    else:
        # SVD, slower, but more accurate
        M, Sing, VT = scipy.linalg.svd(S,\
                               full_matrices=True,\
                               compute_uv=True,\
                               overwrite_a=True)
        # sqrt( 1/(1+sing^2)), addition is elementwise
        Sing = np.sqrt(1/(Sing*Sing + 1))
        STS  = np.dot(VT.T*Sing, VT)
        
    # perform U1 = U0*M + Q*N
    U1 = np.dot((U0+Xi),STS)
    return U1
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# Polar decomposition inverse retraction
# 
# 
# Input arguments      
#          U0 : base point on St(n,p)
#          U1 :      point on St(n,p)
# Output arguments
#          Xi    : tangent vector in T_U0 St(n,p)
#------------------------------------------------------------------------------
def Stiefel_PF_inv_ret(U0, U1):
#------------------------------------------------------------------------------
    # get dimensions
    n,p = U0.shape
    
    M = (-1)*np.dot(U0.T,U1)
    # solve MX + XM = -2*eye(p)
    
    X = scipy.linalg.solve_sylvester(M, M.T, (-2)*np.eye(p,p))
    
    Xi = U1.dot(X) - U0
        
    return Xi
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# Polar light retraction
# PL_U0 (Xi ) = (U0exp(U0'*Xi) + (I-U0U0')*Xi)(I + XiT(I-U0U0')Xi )^−1/2
# 
# Input arguments      
#          U0    : base point on St(n,p)
#          Xi : tangent vector in T_U0 St(n,p)
# Output arguments
#          U1    : PL_U0(Xi)
#------------------------------------------------------------------------------
def Stiefel_PL_ret(U0, Xi, mode=1):
#------------------------------------------------------------------------------
    # get dimensions
    n,p = U0.shape
    
    A = np.dot(U0.T,Xi)  # horizontal component
    
    
    if mode == 1:
        # general purpose matrix exponential
        # seems to be faster than homebrew Schur exp
        expA_A  = scipy.linalg.expm(A) - A
    elif mode == 2:
        # Cayley trafo
        expA_A  = StAux.Cayley(A) - A
    else:
        # eigenvalue decomposition
        # to-do: make this real Schur form
        Lambda, N = linalg.eig(A)

        D  = np.exp(Lambda) - Lambda
        expA_A = np.dot(N*D, N.conj().T)
        expA_A = expA_A.real # output must be real
   
        
        
    # assemble A
    U1      = np.dot(U0, expA_A) + Xi
    
    # compute matrix square root
    # QR of (I-U0U0^T)Xi = Xi - U0*A
    S    = np.linalg.qr(Xi-U0.dot(A), mode='r')   
    
    fast = 1
    if fast:
        # symmetric EVD 
        Lambda, V  = scipy.linalg.eigh(np.dot(S.T, S))
        # sqrt( 1/(1+lambda)), addition is elementwise
        Lambda = np.sqrt(1/(Lambda + 1))
        STS = np.dot(V*Lambda, V.T)
    else:
        # SVD, slower, but more accurate
        M, Sing, VT = scipy.linalg.svd(S,\
                               full_matrices=True,\
                               compute_uv=True,\
                               overwrite_a=True)
        # sqrt( 1/(1+sing^2)), addition is elementwise
        Sing = np.sqrt(1/(Sing*Sing + 1))
        STS  = np.dot(VT.T*Sing, VT)
   
    # assemble U1
    U1 = U1.dot(STS)
    return U1
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Polar light inverse retraction
# PL_inv_U0 (U1) = (U0(logm(MR') -MR') + URS^-1R'
# where MSR' = U0'*U1 (SVD)
# Input arguments      
#          U0  : base point on St(n,p)
#          U1  : end point
# Output arguments
#          Xi  : tangent vector
#------------------------------------------------------------------------------
def Stiefel_PL_inv_ret(U0, U1, mode=1):
    # get dimensions
    n,p = U0.shape
    
    M, S, RT = scipy.linalg.svd(np.dot(U0.T,U1),\
                               full_matrices=True,\
                               compute_uv=True,\
                               overwrite_a=True)
            
    MRT = M.dot(RT)
    
    RinvSRT = np.dot( (RT.T*(1./S)), RT)

    # assemble Xi
    if mode == 1:
        Xi = U0.dot((linalg.logm(MRT) - MRT)) + U1.dot(RinvSRT)
    else:
        # Cayley trafo for replacing the logm
        Xi = U0.dot((StAux.Cayley_inv(MRT) - MRT)) + U1.dot(RinvSRT)
    return Xi