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
import time 


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

#------------------------------------------------------------------------------
# QR retraction
# R_U0(xi) = qf(U0 + xi), where qf() is the Q-factor of U0 + xi == QR,
#                                                              (QR) 
#
# We will compute the UNIQUE QR decomposition, which exist for full-rank 
# inputs. It is made unique by enforcing a positive diagonal on R. 
#
# Input arguments      
#          U0    : base point on St(n,p)
#          Xi : tangent vector in T_U0 St(n,p)
# Output arguments
#          U1    : R_U0(xi)
#------------------------------------------------------------------------------
def Stiefel_QR_ret(U0, Xi):
    Q,R = np.linalg.qr(U0 + Xi,mode='reduced')

    S = np.sign(np.diag(R))
    Q = Q * S
    R = S * R
    
    return Q, R

#------------------------------------------------------------------------------
# QR inverse retraction
# qf^(-1)_U0(U1) = Xi.
#
# Based on Kaneko et.al. 
# "Empirical Arithmetic Averaging Over the Compact Stiefel Manifold", 2013
#
# Input arguments      
#          X  : base point on St(n,p)
#          Q  : end point
# Output arguments
#          Xi  : tangent vector
#------------------------------------------------------------------------------
def Stiefel_QR_inv_ref(X,Q):
    n,r = X.shape
    M = X.T @ Q     # (r x r) matrix
    R = np.zeros(M.shape)
    if M[0,0] > 0:  # Otherwise the unique QR fails to exist
        R[0,0] = 1/M[0,0]
    else:
        print("The inverse QR retraction could not be computed")
        print("Return 0")
        return 0
    
    for i in range(1,r):
        tildeM = M[0:i+1,0:i+1] # The i'th principal minor of M
        
        if np.linalg.det(tildeM) > 1.0e-12:
            b = np.zeros([i+1,1]) # has length i + 1,
            
            for j in range(i+1): # at run 1: j = 0, 1
                if j < i: # j = 0, 1, .. , i-1
                    # print(i,j, tildeM.shape)
                    b[j] = -1 * M[i,0:j+1] @ R[0:j+1, j]
                else: # j = i
                    b[j] = 1
           
            # Solve the linear system 
            tilder = np.linalg.solve(tildeM, b)
            # print(tilder.shape)
            
        else: # tildeM is not invertible 
            print("------")
            print("The ",i,'\'th principal minor is singular. Return 0')
            print("Determinant of tildeM: ", np.linalg.det(tildeM) )
            print("------")
            return 0
        if tilder[i] <= 0:
            print("Non-positive diagonal entry on r occoured. Return 0")
            return 0
        for k in range(0,i+1):
            R[k,i] = tilder[k,0]
        #print(np.diag(R))
    # Error with indices somewhere above, 
    Xi = Q @ R - X
    return Xi

#------------------------------------------------------------------------------
# Quasi geodesic, Grassmann-like
# R_U0(xi) = [U Q] expm(t (A -B^T; B 0 )) * Ip, A,B are p x x matrices.
#                                                              
#
# Based on Bendokat et.al. "Eﬃcient Quasi-Geodesics on the Stiefel Manifold"
#
# Input arguments      
#          U0    : base point on St(n,p)
#          Xi : tangent vector in T_U0 St(n,p)
# Output arguments
#          U1    : R_U0(xi)
#------------------------------------------------------------------------------
def Stiefel_Quasi_geod(U,Xi):
    A = U.T @ Xi
    UperpB = Xi - U @ A
    Q, S, VT = linalg.svd(UperpB, full_matrices=False,compute_uv=True, overwrite_a=True)

    Y = (U @ (VT.T @ np.diag(np.cos(S))) + Q @ np.diag(np.sin(S))) @ (VT @ linalg.expm(A))
    return Y

#------------------------------------------------------------------------------
# Inverse Quasi geodesic
# R_inv_U0(U1) = xi 
#                                             
# Based on Bendokat et.al. "Eﬃcient Quasi-Geodesics on the Stiefel Manifold"
# Algorithm 1 
#
# Input arguments      
#          U0    : base point on St(n,p)
#          Xi : tangent vector in T_U0 St(n,p)
# Output arguments
#          U1    : R_U0(xi)
#------------------------------------------------------------------------------
def Stiefel_inv_Quasi_geod(U0,U1,mode = 1):
    if mode == 1: # Grassmann-like
        Q, S, RT = linalg.svd(U1.T@U0)
        R = Q @ RT
        Ustar = U1 @ R
        A = linalg.logm(R.T)
        Q,S,VT = linalg.svd(Ustar - U0 @ (U0.T @ Ustar), full_matrices=False,compute_uv=True, overwrite_a=True)
        Sig = np.asin(S)

        Xi = U0 @ A + Q @ (Sig * VT.T).T
        return Xi    
    
    elif mode == 2: # Short-econ geod. with Cayley
        tQ, tS, tRT = linalg.svd(U1.T@U0)
        R = tQ @ tRT
        a = linalg.logm(R.T)
        Ustar = U1 @ R

        Q, S,VT = linalg.svd(Ustar - U0 @ (U0.T @ Ustar), full_matrices=False,compute_uv=True, overwrite_a=True)

        Sig = np.asin(S)

        SinS = np.sin(np.copy(Sig))
        CosS = np.cos(np.copy(Sig))

        Sig = np.diag(Sig)

        b = Sig @ VT
        c = -1/6 * b @ a @ b.T

        
        # Return A,B and C
        T = np.concatenate(((VT.T * CosS)  @ VT @ R.T, (-VT.T * SinS) @ linalg.expm(c)),axis = 1)
        L = np.concatenate(((SinS * VT.T).T @ R.T, (CosS * linalg.expm(c).T).T),axis=1)

        ABTBC = StAux.Cayley_inv(np.concatenate((T,L),axis =0))
        w,p = ABTBC.shape

        p = int(p/2)

        A = ABTBC[0:p,0:p]
        B = ABTBC[p:2*p,0:p]
        mBT = ABTBC[0:p,p:2*p]
        C = ABTBC[p:2*p,p:2*p]

        return A,B,mBT,C,ABTBC,Q
    else: # Short-econ geod
        tQ, tS, tRT = linalg.svd(U1.T@U0)
        R = tQ @ tRT
        a = linalg.logm(R.T)
        Ustar = U1 @ R

        Q, S,VT = linalg.svd(Ustar - U0 @ (U0.T @ Ustar), full_matrices=False,compute_uv=True, overwrite_a=True)

        Sig = np.asin(S)

        SinS = np.sin(np.copy(Sig))
        CosS = np.cos(np.copy(Sig))

        Sig = Sig

        b = (Sig * VT.T).T
        c = -1/6 * b @ a @ b.T

        
        # Return A,B and C
        T = np.concatenate(((VT.T * CosS) @ VT @ R.T, (-VT.T * SinS) @ linalg.expm(c)),axis = 1)
        L = np.concatenate(((SinS * VT.T).T @ R.T, (CosS * linalg.expm(c).T).T),axis=1)

        ABTBC = linalg.logm(np.concatenate((T,L),axis =0))
        w,p = ABTBC.shape

        p = int(p/2)

        A = ABTBC[0:p,0:p]
        B = ABTBC[p:2*p,0:p]
        mBT = ABTBC[0:p,p:2*p]
        C = ABTBC[p:2*p,p:2*p]


        return A,B,mBT,C,ABTBC,Q


testQR = False
if testQR:
    np.random.seed(345345)
    n = 10000
    p = 300

    A = np.random.rand(n,p)
    U0,R0 = np.linalg.qr(A,mode='reduced')
   
    # Generate a tangent vector at U0
    A = np.random.rand(p,p)
    A = 0.5 * (A.T - A) # Now A is p x p skew
    Xi = U0 @ A
    Xi = Xi / np.linalg.norm(Xi,'fro') # Normalize for stability 

    #print(np.linalg.norm(Xi.T@U0 + U0.T@Xi))
    U1,R = Stiefel_QR_ret(U0,Xi)
    flag_Is_Stiefel = False
    if np.linalg.norm(U1.T @ U1 - np.eye(p)) < 1e-13:
        flag_Is_Stiefel = True

    flag_Pos_diag = True
    if any(np.diag(R) < 0) == True: # then there is at least one nonpos. el.
        flag_Pos_diag = False
    
    print("*** Retraction checks ***")
    print("U1 = qf(U0 + Xi) is Stiefel? ", flag_Is_Stiefel)
    print("The diagonal of R is strictly positive? ", flag_Pos_diag)
    print("***                  ***")

    Xi_inv = Stiefel_QR_inv_ref(U0,U1)
    print("*** Inverse retraction checks ***")
    print("Norm of true xi: ",np.linalg.norm(Xi))
    print("Norm of retrieved inverse: ", np.linalg.norm(Xi_inv))
    print("Difference of inverse and true tangent: ",np.linalg.norm(Xi_inv-Xi))
    

testQuasi = False
if testQuasi:
    np.random.seed(345345)
    n = 10000
    p = 300

    A = np.random.rand(n,p)
    U0,R0 = np.linalg.qr(A,mode='reduced')
   
    # Generate a tangent vector at U0
    A = np.random.rand(p,p)
    A = 0.5 * (A.T - A) # Now A is p x p skew
    Xi = U0 @ A
    Xi = Xi / np.linalg.norm(Xi,'fro') # Normalize for stability 

    U1 = Stiefel_PL_ret(U0,Xi)
    U1 = Stiefel_Quasi_geod(U0,Xi)
    flag_Is_Stiefel = False
    if np.linalg.norm(U1.T @ U1 - np.eye(p)) < 1e-13:
        flag_Is_Stiefel = True
    print("*** Retraction checks ***")
    print("U1 = R_U0(Xi) is Stiefel? ", flag_Is_Stiefel)
    print("***                  ***")

    Xi_inv = Stiefel_inv_Quasi_geod(U0,U1)
    print("*** Inverse retraction checks ***")
    print("Norm of true xi: ",np.linalg.norm(Xi))
    print("Norm of retrieved inverse: ", np.linalg.norm(Xi_inv))
    print("Difference of inverse and true tangent: ",np.linalg.norm(Xi_inv-Xi))