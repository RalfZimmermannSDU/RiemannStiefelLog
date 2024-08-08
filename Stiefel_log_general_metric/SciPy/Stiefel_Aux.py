#------------------------------------------------------------------------------
#@author: Ralf Zimmermann, IMADA, SDU Odense
# zimmermann@imada.sdu.sdk
#
# This file implements the following methods
# that are mainly auxiliaries for the Stiefel Exp and Log methods
#
# For theoretical background, see:
#
# R. Zimmermann, K. H"uper.
# "Computing the Riemannian logarithm on the Stiefel manifold: 
#  metrics, methods and performance", arXiv:2103.12046, March 2022
#  https://arxiv.org/abs/2103.12046
#
#
# * alphaMetric(D1, D2, U, metric_alpha):
#       - compute inner product: <D1, D2>_U  w.r.t. the alpha metric
#
# * Exp4Geo(A, R, metric_alpha):
#       - evaluate eq. (12) of Zimmermann/H"uper.
#
# * Exp4Geo_pre(A, R, metric_alpha):
#      - evaluate eq. (12) of Zimmermann/H"uper based on precomputed EVD
#
# * Stiefel_approx_parallel_trans_p(M2, N2, A1, R1, nu):
#      - implements Algorithm 3 of Zimmermann/H"uper
#
# * Cayley(X):
#      - classical Cayley transformation
#
# * solvsymsyl(A, C):
#      - solve the sylvester equation AX + XA = C, with A sym, C skew
#
# * A2skew(A):
#      - compute 0.5*(A-A^T)
#
# * A2sym(A):
#      - compute 0.5*(A+A^T)
#------------------------------------------------------------------------------

import scipy
import numpy as np

#------------------------------------------------------------------------------
# alpha-metric on T_U St(n,p)
#
# input arguments
#          D1, D2 = tangent vectors
#               U = base point
#    metric_alpha = metric parameter
# output arguments
#      Riem_inner = <D1, D2>_U
#------------------------------------------------------------------------------
def alphaMetric(D1, D2, U, metric_alpha=0.0):
#------------------------------------------------------------------------------
    a1 = np.trace(np.dot(D1.T, D2))
    a2 = np.trace(np.dot(np.dot(D1.T, U), np.dot(U.T, D2)))
    
    if abs(metric_alpha + 1.0) >1.0e-13:
        x  = (2*metric_alpha + 1)/(metric_alpha + 1)
    else:
        print('Wrong metric parameter in <alphaMetric>')
        x  = 1.0
    
    Riem_inner = a1 - (0.5*x)*a2
    return Riem_inner
#------------------------------------------------------------------------------




#------------------------------------------------------------------------------
# helper function
#
# evaluate the matrix exponential as inherent to the Stiefel geodesics
#
# unified implementation for all alpha-metrics 
#
#
# input arguments
#               A = matrix factor in candidate matrix U*A+Q*R
#               R = matrix factor in candidate matrix U*A+Q*R
#    metric_alpha = metric parameter
# output arguments
#   M = matrix factor, upper block               | xA  -R^T  |   |expm(yA|
#   N = matrix factor, lower block of matrix expm| R^T   0   | * |   0  |
#------------------------------------------------------------------------------
def Exp4Geo(A, R, metric_alpha):
#------------------------------------------------------------------------------
    p = A.shape[0]
    if abs(metric_alpha) < 1.0e-13:                          # canonical metric
        # build block matrix
        upper = np.concatenate((A, -R.transpose()), axis=1)
        lower = np.concatenate((R, np.zeros((p,p), dtype=R.dtype)),axis=1)
        L     = np.concatenate((upper, lower), axis=0)
        V = scipy.linalg.expm(L)
        M = V[0:p,0:p]
        N = V[p:2*p,0:p] 
    elif abs(metric_alpha+1.0) > 1.0e-13:
        # metric factors
        x = 1.0/(metric_alpha +1)
        y  = (metric_alpha)/(metric_alpha +1)
        
        upper = np.concatenate((x*A, -R.transpose()), axis=1)
        lower = np.concatenate((R, np.zeros((p,p), dtype=R.dtype)),axis=1)
        L     = np.concatenate((upper, lower), axis=0)
        V     = scipy.linalg.expm(L)
        Phi   = scipy.linalg.expm(y*A)
        M     = np.dot(V[0:p,0:p],Phi)
        N     = np.dot(V[p:2*p,0:p],Phi)
    else:
        print('Error in  Stiefel_Exp: wrong metric. Choose alpha != -1.')
        M = 0
        N = 0

    return M,N
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# helper function
#
# evaluate the matrix exponential as inherent to the Stiefel geodesics
#
# unified implementation for Euclidean and canonical metric
#
# This function expects the EVD 
# Evecs*evals*Evecs' = [[f(A), -R'];[R, 0]] as an input.
# Here, f(A)=A_pre depends on the chosen alpha-metric
#------------------------------------------------------------------------------
def Exp4Geo_pre(t, A_pre, Evecs, evals, metric_alpha):
#------------------------------------------------------------------------------
    p = A_pre.shape[0]
    
    #evaluate the matrix exponential, the result must be real
    #compute "diag x matrix" elementwise
    # it turns out that the way, scipy use broadcast yields
    #  diag(d)* X.conj().T = (X*d.conj().T).conj().T
    
    V = Evecs[0:p,0:2*p] * np.exp((-t*1j)*evals)
    V = np.dot(Evecs, V.conj().T)
    V = np.real(V)
    
    if abs(metric_alpha) < 1.0e-13:                           # canonical metric
        M = V[0:p,0:p]
        N = V[p:2*p,0:p]
    elif abs(metric_alpha+1.0) > 1.0e-13:
        # If A_pre = (2-x)*A, where x = (2*alpha + 1)/(alpha + 1)
        # then we need y*A,   where y  = (alpha)/(alpha +1)
        # which is given by (y/(1-y))*A_pre = alpha*A_pre    
        Phi = scipy.linalg.expm((t*metric_alpha)*A_pre)
        M = np.dot(V[0:p,0:p],Phi)
        N = np.dot(V[p:2*p,0:p],Phi)
    else:
        print('Error in  Stiefel_Exp: wrong metric. Choose alpha != -1.')
        M = 0 
        N = 0
    return M,N
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# Stiefel_approx_parallel_trans_p
#
# Given Y1,Y2 in St(n,p) and Delta in T_Y1St(n,p),
# approximate the parallel transport of Delta to 
# Delta2 in T_Y2St(n,p)
#
# SPECIAL CASE, WHERE Y1, Y2, Delta can be represented as
#
#    Y1 = U*M1 + Q*N1
#    Y2 = U*M2 + Q*N2U
# Delta = U*A1 + Q*R1
#
# WITH THE SAME FIXED U, Q!
#
# Input: 
# Y2    = U*M2 + Q*N2   : in St(n,p)
# Delta = U*A + Q*R     : in T_Y1 St(n,p)
# nu    = norm of Delta( to be conserved)
#
# Output:
# Delta2 = U*A2 + Q*R2  : in T_Y2 St(n,p)
#------------------------------------------------------------------------------
def Stiefel_approx_parallel_trans_p(M2, N2, A1, R1, nu):
#------------------------------------------------------------------------------
    epsi = 1.0e-14

    # project Delta onto T_Y2 St(n,p) using the pxp factors only
    sym_part = A2sym( np.dot(M2.T,A1) + np.dot(N2.T,R1))

    A2 = A1 - np.dot(M2,sym_part)
    R2 = R1 - np.dot(N2,sym_part)
    # rescale  
    # note: trace(A.T*A) = <A(:), A(:)> 
    
    l = np.sqrt(np.dot(A2.flatten(), A2.flatten())\
                   + np.dot(R2.flatten(), R2.flatten()))

    if l > epsi:
        A2 = (nu/l)*A2
        R2 = (nu/l)*R2
    else:
        A2 = np.zeros(A1.shape)
        R2 = np.zeros(A1.shape)
        print("para trans zero case, nu=", nu)

    return A2, R2
#------------------------------------------------------------------------------


#--------------------------------------------------------------------------
#
# local function: efficient Schur log evaluation
#
# Compute matrix function logm(V) of ORTHOGONAL matrix via Schur decomposition
# Caution It returns only those blocks that are needed in Stiefel log.
#
# Inputs:
#      V : real orthogonal matrix
# Outputs:
#   logV :
#--------------------------------------------------------------------------
def SchurLog(V):
    # get dimensions
    n = V.shape[0]

    flag_negval = 0  # raise a flag, if there is a negative real eigenvalue

    # start with real Schur decomposition Q S Q^T
    S, Q = scipy.linalg.schur(V)
    # S must have 2x2-block diagonal form
    # create empty sparse matrix
    logS = scipy.sparse.lil_matrix((n,n))
    k = 0
    # compute log of 1x1 and 2x2 blocks
    while k < n:
        # is block of dim 1x1 => real eigenvalue? 
        if k==n-1:
            if abs(S[k,k] +1.0)<1.0e-13:
                print('Error: negativ eigval on real axis')
                flag_negval = 1
            k = k+1
        elif abs(S[k+1,k])<1.0e-13:
            if abs(S[k,k] +1.0)<1.0e-13:
                print('Error: negativ eigval on real axis')
                flag_negval = 1
            # entry stays zero, just skip ahead
            k = k+1
        else:
            # there is a 2x2 block S(k:k+1, k:k+1)
            z = S[k,k] + 1j*S[k,k+1]
            phi = np.log(z).imag
            logS[k,k+1] =  phi
            logS[k+1,k] = -phi
            k=k+2
    # end while
    
    # form log matrix
    logV = np.dot(Q, logS.dot(Q.T))
    return logV, flag_negval



#------------------------------------------------------------------------------
# Cayley trafo of X:
# Cayley(X) = (I - 0.5X)^{-1}*(I + 0.5X)
#------------------------------------------------------------------------------
def Cayley(X):
#------------------------------------------------------------------------------
    p = X.shape[0]
    # diagonal indicces
    diag_pp = np.diag_indices(p)
    # form I-0.5X
    Xminus = -0.5*X
    Xminus[diag_pp] = Xminus[diag_pp] + 1.0
    # form I+0.5X
    Xplus  = 0.5*X
    Xplus[diag_pp] = Xplus[diag_pp] + 1.0
    
    Cay = np.dot(np.linalg.inv(Xminus), Xplus)
    return Cay
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# solve the symmetric sylvester equation
# AX + XA  = C
#
# with A real, symmetric, C real, skew-symmetric
#
# via Bathia, Matrix Analysis, Theorem VII.2.3, p. 205
#------------------------------------------------------------------------------
def solvsymsyl(A, C):
    #
    # 
    #
    #
    # step 0
    try:
        # local C-Modul
        import C_matrix_ops_swig.C_matrix_ops as C_matrix_ops
    except ImportError:
        C_matrix_ops = None
    
    # step 1: reduce to diagonal problem A = Q L Q'
    #
    # AX + XA = C  <=> L Q'XQ + Q'XQ L = Q'C Q
    L, Q = np.linalg.eigh(A)
    #
    C2 = np.dot(Q.T, np.dot(C,Q))
     # step 2: build solution matrix
    n = C.shape[0]
    if C_matrix_ops != None:
        print("execute C code")
        X = np.zeros((n*n,))
        C_matrix_ops.symsylv_buildsolmat_func(C2.flatten(), L, X, n)
        X = X.reshape((n,n))
    else:
        X = np.zeros((n,n))   
        print("execute python code")
        for j in range(n):
            for k in range(j+1,n):
                X[j,k] = C2[j,k]/(L[j]+L[k])
                X[k,j] = -X[j,k]

    X = np.dot(Q, np.dot(X, Q.T))
    #make X exactly skew
    X =A2skew(X)
    return X
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
def A2skew(A):
#------------------------------------------------------------------------------
    # extract the skew-symmetric part of A
    Askew = 0.5*(A-A.T)
    return Askew
#------------------------------------------------------------------------------
    

#------------------------------------------------------------------------------
def A2sym(A):
#------------------------------------------------------------------------------
    # extract the symmetric part of A
    Asym = 0.5*(A+A.T)
    return Asym
#------------------------------------------------------------------------------
