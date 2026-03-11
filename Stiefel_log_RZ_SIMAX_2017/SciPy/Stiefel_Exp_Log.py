#------------------------------------------------------------------------------
#@author: Ralf Zimmermann, IMADA, SDU Odense
# zimmermann@imada.sdu.sdk
# This file implements the following methods:
#
# * Stiefel_Exp(U0, Delta):
#       - compute U1 = Exp^{St}_U0(Delta)
#
# * Stiefel_Log_alg(U0, U1, tau, do_Procrustes=0, do_Cayley=0, do_Sylvester=1):
#       - compute Delta = Log^{St}_U0(U1), up to an accuracy of tau
#         via algebraic Stiefel log for the canonical metric
#
# * Cayley(X): Cayley transformation 
#
# * canonicalMetric(D1, D2, U):
#       - compute canonical inner product: <D1, D2>_U
#
# * distStiefel(U1, U2):
#       - compute dist(U1,U2) = ||Log_U1(U2)||
#
# * create_random_Stiefel_data(n, p, dist):
#       - create (pseudo-)random Stiefel data: for testing purposes
#
#------------------------------------------------------------------------------

import numpy
import scipy
from scipy import linalg
from numpy import random


###############################################################################
# FUNCTION DECLARATION
###############################################################################


#------------------------------------------------------------------------------
# Exponential map on Stiefel manifold (anonical metric)
# Following 
#
# A. Edelman, T. A. Arias, and S. T. Smith. 
# "The geometry of algorithms with orthogonality constraints." 
# SIAM Journal on Matrix Analysis and
# Applications, 20(2):303-353, 1998.
#
# Input arguments      
#   U0    : base point on St(n,p)
#   Delta : tangent vector in T_U0 St(n,p)
# Output arguments
#   U1    : Exp^{St}_U0(Delta),
#------------------------------------------------------------------------------
def Stiefel_Exp(U0, Delta):
#------------------------------------------------------------------------------
    # get dimensions
    n,p = U0.shape
    A = scipy.dot(U0.transpose(),Delta)  # horizontal component
    K = Delta-scipy.dot(U0,A)            # normal component
    
    # qr of normal component
    QE, Re = scipy.linalg.qr(K, overwrite_a=True,\
                                lwork=None,\
                                mode='economic',\
                                pivoting=False,\
                                check_finite=True)

    # canonical metric
    # matrix exponential
    upper = scipy.concatenate((A, -Re.transpose()), axis=1)
    lower = scipy.concatenate((Re, scipy.zeros((p,p), dtype=Re.dtype)), axis=1)
    L     = scipy.concatenate((upper, lower), axis=0)
    MNe   = scipy.linalg.expm(L)
    MNe   = MNe[:,0:p]

    # perform U1 = U0*M + Q*N
    U1 = scipy.dot(U0,MNe[0:p,0:p]) +  scipy.dot(QE, MNe[p:2*p,0:p])
    return U1
#------------------------------------------------------------------------------





#------------------------------------------------------------------------------
# Riemannian Stiefel logarithm
# 
# following
# R. Zimmermann 
# "A matrix-algebraic algorithm for the Riemannian logarithm
# on the Stiefel manifold under the canonical metric."
# SIAM Journal on Matrix Analysis and Applications, 38(2):322-342, 2017.
#
# Input arguments      
#  U0, U1 : points on St(n,p)
#     tau : convergence threshold
#
# meta-parameters:
# do_Procrustes = 0/1: Do Procrustes preprocessing?
# do_Cayley     = 0/1: use Cayley trafo to approximate the matrix exp.
#
# Output arguments
#    Delta : Log^{St}_U0(U1), 
#            i.e. tangent vector such that Exp^St_U0(Delta) = U1
#conv_hist : convergence history
#------------------------------------------------------------------------------
def Stiefel_Log_alg(U0, U1, tau, do_Procrustes=0, do_Cayley=0):
#------------------------------------------------------------------------------
    # get dimensions
    n,p = U0.shape
    
    check_det = 1
    # step 1
    M = scipy.dot(U0.T, U1)
    # step 2
    U0orth = U1 - scipy.dot(U0,M) 
    # thin qr of normal component of U1
    Q, N = scipy.linalg.qr(U0orth, overwrite_a=True,\
                                lwork=None,\
                                mode='economic',\
                                pivoting=False,\
                                check_finite=True)
    # step 3
    MN = scipy.concatenate((M,N), axis=0)    
    # orthogonal completion
    V, Rq = scipy.linalg.qr(MN, overwrite_a=True,\
                               lwork=None,\
                               mode='full',\
                               pivoting=False,\
                               check_finite=True)

    if do_Procrustes:
        # "Procrustes preprocessing"
        # SVD of lower diagonal block of V
        D, S, R = scipy.linalg.svd(V[p:2*p,p:2*p],\
                                   full_matrices=False,\
                                   compute_uv=True,\
                                   overwrite_a=False)
        R = R.transpose()
        # apply Procrustes rotation
        V[:,p:2*p] = scipy.dot(V[:,p:2*p], scipy.dot(R,D.T))
        
    V = scipy.concatenate((MN, V[:,p:2*p]), axis=1)
                                           #          |M  X0|
                                           # now, V = |N  Y0| 
    # just for the records
    #norm_logmV = linalg.norm(linalg.logm(V), 2)
    #if numpy.absolute(norm_logmV - numpy.pi) < 1.0e-10:
    #    print('Initial matrix log not well defined: Distance to pi =',\
    #           norm_logmV -numpy.pi)
    
    # check if V \in SO(2p)
    if check_det:
        # ensure that V \in SO(n) 
        if do_Procrustes:
            if numpy.power(-1,p)*numpy.prod(numpy.diag(Rq[:p,:p]))*numpy.linalg.det(numpy.dot(R,D.T)) < 0:
                # flip sign of one column
                V[:,p] = (-1)*V[:,p]
        else:
            if numpy.power(-1,p)*numpy.prod(numpy.diag(Rq[:p,:p])) < 0:
                # flip sign of one column
                V[:,p] = (-1)*V[:,p]
             
    conv_hist = []                                                     
    # step 4: FOR-Loop
    for k in range(1000):
        # step 5
        LV = linalg.logm(V)
                                  # standard matrix logarithm
                                  #             |Ak  -Bk'|
                                  # now, LV =   |Bk   Ck |
        # some safety measures:
        # ensure that LV is real and skew
        numpy.fill_diagonal(LV, 0.0)
        LV = numpy.real(0.5*(LV-LV.T))
        
        C = LV[p:2*p, p:2*p]      # lower (pxp)-diagonal block
 
        # steps 6 - 8: convergence check
        normC = linalg.norm(C, 'fro')
        conv_hist.append(normC)
        if normC<tau:
            print('Stiefel log converged after ', len(conv_hist), ' iterations.')
            break
        
        # step 9
        # exponential of updated block
        if do_Cayley:
            # Cayley approx
            Phi = Cayley(-C)
        else:
            # standard matrix exponential
            Phi = linalg.expm(-C)

        # step 10: rotate the last p columns
        V[:,p:2*p] = scipy.dot(V[:,p:2*p],Phi)   # update last p columns
  
    # prepare output                         |A  -B'|
    # upon convergence, we have  logm(V) =   |B   0 | = LV
    #     A = LV(1:p,1:p);     B = LV(p+1:2*p, 1:p)
    # Delta = U0*A+Q*B
    Delta = scipy.dot(U0,LV[0:p,0:p]) + scipy.dot(Q, LV[p:2*p, 0:p])
    return Delta, conv_hist
#------------------------------------------------------------------------------






#------------------------------------------------------------------------------
# Cayley trafo of X:
# Cayley(X) = (I - 0.5X)^{-1}*(I + 0.5X)
#------------------------------------------------------------------------------
def Cayley(X):
#------------------------------------------------------------------------------
    p = X.shape[0]
    # diagonal indices
    diag_pp = scipy.diag_indices(p)
    # form I-0.5X
    Xminus = -0.5*X
    Xminus[diag_pp] = Xminus[diag_pp] + 1.0
        # form I+0.5X
    Xplus  = 0.5*X
    Xplus[diag_pp] = Xplus[diag_pp] + 1.0
    
    Cay = scipy.dot(scipy.linalg.inv(Xminus), Xplus)
    return Cay
#------------------------------------------------------------------------------





#------------------------------------------------------------------------------
# Riemmannian distance on St(n,p)
#
# Input arguments
#   U1, U2 = points on St(n,p)
#
# Output arguments
#   dist(U1,U2) = || Log_U1(U2)||
#------------------------------------------------------------------------------
def distStiefel(U1, U2):
#------------------------------------------------------------------------------
    tau = 1.0e-13   # numerical threshold for log alg, hard-coded
    Delta, conv = Stiefel_Log_alg(U1, U2, tau)      
    dist = scipy.sqrt(canonicalMetric(Delta, Delta, U1))
    return dist
#------------------------------------------------------------------------------





#------------------------------------------------------------------------------
# create a pseudo-random data set 
#   U0,U1 \in St(n,p), Delta \in T_U St(n,p) such that
#   U1 = Exp_U0(Delta)
# 
# Input arguments
# (n,p) = dimension of the Stiefel matrices
# dist  = Riemannian distance between the points U0,U1
#         that are to be created
# Output arguments: 
# U0, U1 : points on St(n,p), U1 = Exp_U0(Delta)
#  Delta : tangent vector on T_U St(n,p) with canonical norm 'dist',
#          which is also the Riemannian distance dist(U0,U1)
#
#------------------------------------------------------------------------------
def create_random_Stiefel_data(n, p, dist):
#------------------------------------------------------------------------------
    #create pseudo-random stiefel matrix:
    #numpy.random.seed(seed=0)
    X =  random.rand(n,p)
    U0, R = scipy.linalg.qr(X, overwrite_a=True,\
                                lwork=None,\
                                mode='economic',\
                                pivoting=False,\
                                check_finite=True)

    # create pseudo-random tangent vector in T_U0 St(n,p)
    A = random.rand(p,p)
    A = A-A.transpose()   # "random" p-by-p skew symmetric matrix
    T = random.rand(n,p)
    # create Delta = U*A + (I-UU^T)*T
    Delta = scipy.dot(U0,A)+ T-scipy.dot(U0,scipy.dot(U0.transpose(),T))
    #normalize Delta w.r.t. the canonical metric
    norm_Delta = scipy.sqrt(canonicalMetric(Delta, Delta, U0))
    Delta = (dist/norm_Delta)*Delta
    # 'project' Delta onto St(n,p) via the Stiefel exponential
    U1 = Stiefel_Exp(U0, Delta)
    return U0, U1, Delta
#------------------------------------------------------------------------------





#------------------------------------------------------------------------------
# canonical metric on T_U St(n,p)
#
# Input arguments
#   D1, D2 = tangent vectors in T_U St(n,p)
#        U = base point
#
# Output arguments
# Riem_inner = <D1, D2>_U
#------------------------------------------------------------------------------
def canonicalMetric(D1, D2, U):
#------------------------------------------------------------------------------
    a1 = scipy.trace(scipy.dot(D1.T, D2))
    a2 = scipy.trace(scipy.dot(scipy.dot(D1.T, U), scipy.dot(U.T, D2)))

    
    Riem_inner = a1 - 0.5*a2
    return Riem_inner
#------------------------------------------------------------------------------
