#------------------------------------------------------------------------------
#@author: Ralf Zimmermann, IMADA, SDU Odense
# zimmermann@imada.sdu.sdk
#
# This file implements the following methods:
#
# * Stiefel_diff_QR(A0, V):
#            - differentiate the QR decomposition of a matrix curve
#              A(t) =  A0 + t*V
#
#
# * diff_expm(M, dM):
#            - differentiate the matrix exponential along a matrix curve
#              M(t) =  M + t*dM, i.e., compute d/dt|t=0  exp_m( M+t*dM)
#
#
# * Stiefel_diff_exp(U, Delta, V):
#            - differentiate the Stiefel exponential along the curve
#              Delta + t*V in T_U0 St(n,p),
#              i.e., compute  d/dt|t=0  Exp_U1 (Delta + t*V)
#
# * Stiefel_diff_SVD(Y, dY, U, S, V)
#            - differentiate the singular value decomposition
#
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
# Differentiate the QR decomposition
#
# A(t) = Q(t)R(t)
# at t= 0
# dA(0) = dQ(0) R(0) + Q(0)dR(0)
# 
# following Walter, Lehmann, Lamour: "On evaluating higher order derivatives
# of the QR decomposition ...."
# Optimization Methods and Software 27:2, p 391-403, 2012
#
# see in particular Prop. 2.2
#
# Here: for a linear matrix function: A(t) = A0 + tV
#
# Inputs: 
#    A0  : (nxp) base point matrix
#    V   : (nxp) velocity 
#
#------------------------------------------------------------------------------
def Stiefel_diff_QR(A0, V):
#------------------------------------------------------------------------------
    # get dimensions
    n,p = A0.shape
    
    # qr of base matrix A0
    Q, R = linalg.qr(A0, overwrite_a=True,\
                               lwork=None,\
                               mode='economic',\
                               pivoting=False,\
                               check_finite=True)
    # invert R
    Rinv = linalg.solve_triangular(R, numpy.eye(p))
    # clean up numerically
    for i in range(Rinv.shape[0]):
        for j in range(i):
            Rinv[i,j] = 0.0     
    # construct helper matrix D = Q^T dQ 
    QTV = numpy.dot(Q.transpose(),V)
    D = numpy.dot(QTV, Rinv) 
    D = select_lower(D)
    # make D skew
    D = D - D.transpose()
    
    # compute derivative of R
    dR = QTV - numpy.dot(D, R)
    # clean up numerically
    for i in range(dR.shape[0]):
        for j in range(i):
            dR[i,j] = 0.0    
    # compute derivative of Q
    dQ = V - numpy.dot(Q, QTV)
    dQ = numpy.dot(dQ, Rinv) +  numpy.dot(Q,D)

    return Q, dQ, R, dR
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# return lower triangle of M
#------------------------------------------------------------------------------
def select_lower(M):

        # get dimensions
    n,p = M.shape
    
    L = numpy.zeros((n,n))
    
    for i in range(n):
        for j in range(i):
            L[i,j] = M[i,j]
    
    return L
#------------------------------------------------------------------------------
    

#-----------------------------------------------------------------------------
# Differentiate the matrix exponential along the curve
# M + t*dM at t=0
# i.e., compute
#
# d/dt|t=0  exp_m( M+t*dM)
# 
# Use Mathias Theorem, see
# N. Higham, "Functions of Matrices" Theorem 3.6, p. 58
#
# Nice aside: produces exp(M) and d exp(M)
#
#------------------------------------------------------------------------------
def diff_expm(M, dM):
#------------------------------------------------------------------------------
    n,p =  M.shape   
    #Aux = scipy,zeros((2*n,2*n))
    upper = numpy.concatenate((M, dM), axis=1)
    lower = numpy.concatenate((numpy.zeros((n,n)), M), axis=1)
    Aux   = numpy.concatenate((upper, lower), axis=0)
    # apply the matrix exp
    dexpm = linalg.expm(Aux)

    # pick the corresponding sub-blocks
    expM  = dexpm[0:n, 0:n]
    dexpM = dexpm[0:n, n:2*n]
    
    return expM, dexpM
#----------------------------------------------------------------------------


#----------------------------------------------------------------------------
# Differentiate the Stiefel exponential along the curve
# Delta + t*V in T_U0 St(n,p),
# i.e., compute
#
# d/dt|t=0  Exp_U1 (Delta + t*V)
#------------------------------------------------------------------------------
def Stiefel_diff_exp(U, Delta, V, metric_alpha=0.0):
#------------------------------------------------------------------------------
    # get dimensions
    n,p = U.shape
    
    #  *************************************
    # step 0: preparations
    #project onto orthogonal complement
    # X0 = (I-UU^T)Delta,  Y0 = (I-UU^T)V
    #  *************************************
    A0 = numpy.dot(U.transpose(), Delta)
    #enforce skew-symmetry
    A0 = 0.5*(A0-A0.T)
    X0 =  Delta - numpy.dot(U, A0)
    # derivative d/dt (U.T * (Delta + tV))
    dA0 = numpy.dot(U.transpose(), V)
    #enforce skew-symmetry
    dA0 = 0.5*(dA0-dA0.T)
    Y0 = V - numpy.dot(U, dA0)   
    
    #differentiate qr-decomp. of X0+ t*Y0
    Q, dQ, R, dR = Stiefel_diff_QR(X0, Y0)
    
    #****************************************
    #step 1: differentiating the matrix exp
    #assemble the skew argument matrices
    #****************************************
    
    # metric_factor
    v = 1.0/(metric_alpha +1.0)
    # construct M block matrix
    upper = numpy.concatenate((v*A0, -R.transpose()), axis=1)
    lower = numpy.concatenate((R, numpy.zeros((p,p))), axis=1)
    M     = numpy.concatenate((upper, lower), axis=0)
    #
    # construct dM block matrix
    upper = numpy.concatenate((v*dA0, -dR.transpose()), axis=1)
    lower = numpy.concatenate((dR, numpy.zeros((p,p))), axis=1)
    dM     = numpy.concatenate((upper, lower), axis=0)
    # differentiate the matrix exp
    # bonus: the matrix exp of M is computed simultaneously
    # both are needed in the following
    expM, dexpM = diff_expm(M, dM)
    
    # reduce to the first p-block
    expM_I_0  =  expM[:, 0:p]
    dexpM_I_0 = dexpM[:, 0:p]
    
    # for all alpha metrics except the canonical, we need the smaller
    # matrix-exp as well
    if abs(metric_alpha)>1.0e-10:
        mu = metric_alpha/(metric_alpha +1.0)
        expA, dexpA = diff_expm(mu*A0, mu*dA0)
    

    #********************************************
    #step2: assemble the Riemann-Exp. derivative
    #********************************************
    if abs(metric_alpha)>1.0e-10 and abs(metric_alpha+1.0)>1.0e-10:
        # case: alpha-metric, alpha \neq 0
        # compute
        #    (0, dQ) expm(M)  (exp(A) ;0) 
        #   +(U,  Q) dexpm(M) (exp(A) ;0)
        #   +(U,  Q) expm(M)  (dexp(A);0)       
        dSt_Exp = numpy.dot(dQ, numpy.dot( expM_I_0[p:2*p, 0:p], expA))\
                + numpy.dot(U,  numpy.dot(dexpM_I_0[0:p,   0:p], expA))\
                + numpy.dot(Q,  numpy.dot(dexpM_I_0[p:2*p, 0:p], expA))\
                + numpy.dot(U,  numpy.dot( expM_I_0[0:p,   0:p],dexpA))\
                + numpy.dot(Q,  numpy.dot( expM_I_0[p:2*p, 0:p],dexpA))
    else:
        # case: canonical metric
        # compute
        # (0, dQ) expm(M) (I;0) + (U,Q)*dexpm(M) (I;0)
        dSt_Exp = numpy.dot(dQ, expM_I_0[p:2*p, 0:p])\
                + numpy.dot(U, dexpM_I_0[0:p, 0:p])\
                + numpy.dot(Q, dexpM_I_0[p:2*p, 0:p])
                
    return dSt_Exp
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#differentiate singular value decomposition
#
# Input:
#  Y : snapshot matrix of dim nxm
# dY : derivative of snapshots of dim nxm
# USV' = Y (SVD), S stored as a vector
# REMARK: expect truncated SVD of the form
#  U S V^T
#     U: nxp
# Sigma: as a p-vector
#     V: mxm
# FULL ORTH. MATRIX V IS NEEDED FOR CORRECT DERIVATIVES dV
# BUT NOT NEEDED FOR dU, dS
#
#
#
# Output:
# dU     : derivative, tangent vector in T_U St(n,p)
# dU_bot : orth. component of dU
# A      : = dU'*U
# Alpha  : skew-symmetric matrix of dV = V*Alpha
# dS     : derivatives of singular values
# dV     : derivative of V, tangent vector in T_V St(m,p)
#------------------------------------------------------------------------------
def Stiefel_diff_SVD(Y, dY, U, S, V):
    # get dimensions    
    n,p = U.shape
    m = V.shape[0]
    
    # reciprocal singular values
    S_inv = 1.0/S
    # derivatives of singular values
    dS = numpy.zeros((p,))
    for k in range(p):
        dS[k] = numpy.dot(U[:,k].T, numpy.dot(dY, V[:,k]))[0,0]

    # normal component of dU
    # compute V * S^(-1) = V \odot Ones * sinv_vec
    #
    VSinv     = numpy.dot(V[:,0:p], numpy.diag(S_inv))
    dYVSinv   = numpy.dot(dY, VSinv)
    UTdYVSinv = numpy.dot(U.T, dYVSinv)
    dU_bot    = dYVSinv - numpy.dot(U, UTdYVSinv)

    # tangential component requires derivative of V:
    # it holds: dV = V*Alpha,  Alpha skew
    #
    #dYTY = numpy.dot(dY.T, Y) + numpy.dot(Y.T,dY)
    UTdY = numpy.dot(U.T, dY)


    # assemble Alpha without explicitely computing d/dt (Y'*Y)
    # first pxp block
    Alpha_p = numpy.zeros((p,p))
    # comput U'*dY*V_p
    C     = numpy.dot(UTdY, V[:,0:p])
    
    # upper pxp diagonal block
    for j in range(p): #j=1:p
        for k in range(j+1,p):
            Alpha_p[j,k] = (S[j]*C[j,k] + S[k]*C[k,j]) / ((S[k]+S[j])*(S[k] - S[j]))
            Alpha_p[k,j] = -Alpha_p[j,k]
            # lower (m-p) x p block, approx s(j) = 0 for j>p+1
    
    dV = numpy.dot(V[:,0:p], Alpha_p)
    # Check, if full V is available, i.e., V has m columns

    if (m == V.shape[1]) and (m>p):
        Alpha_mp = numpy.zeros((m-p,p))
        # comput U'*dY*V_p
        C     = numpy.dot(UTdY, V[:,p:m])
        for j in range(m-p):
            for k in range(p):
                Alpha_mp[j,k] = C[k,j]/S[k]
        # add components
        dV = dV + numpy.dot(V[:,p:m], Alpha_mp)
    
    # return only upper pxp-blockof Alpha
    Alpha = Alpha_p
    # compute A = U'*dU
    X = numpy.dot(numpy.dot(numpy.diag(S) , Alpha) - numpy.diag(dS), numpy.diag(S_inv))

    A = UTdYVSinv + X
    # guarantee skew-symmetry
    A = 0.5*(A-A.T)

    dU = dYVSinv + numpy.dot(U,X)

    return dU_bot, A, dU, Alpha, dS, dV
    #------------------------------------------------------------------------------



# test the QR-derivative:
# check if V = dQ*R +Q*dR
test_QRdiff = 0
if test_QRdiff:
    A = random.rand(10000, 200)
    V = random.rand(10000, 200)
    
    Q, dQ, R, dR = Stiefel_diff_QR(A, V)
    # does it match the derivative of A+tV:
    dTest = numpy.dot(dQ, R) + numpy.dot(Q, dR)
    norm_check_diffQR = linalg.norm(dTest-V)/linalg.norm(V, 'fro')
    if norm_check_diffQR < 1.0e-12:
        print('QR-diff worked!')
        print('norm_check_diffQR=', norm_check_diffQR)
    else:
        print('QR-diff did not work!')
        print('norm_check_diffQR=', norm_check_diffQR)

    # Check the expm derivative
    T1 = diff_expm(R, dR)



# test the SVD-derivative:

test_SVDdiff = 0
if test_SVDdiff:
    Y  = random.rand(1000, 30)
    dY = random.rand(1000, 30)
    
    U, S, VT = scipy.linalg.svd(Y,\
                                full_matrices=False,\
                                compute_uv=True,\
                                overwrite_a=False)
    V = numpy.matrix(VT.T)
    
    
    dU_bot, A, dU, Alpha, dS, dV = Stiefel_diff_SVD(Y, dY, U, S, V)
    # does it match the derivative of A+tV:
    S = numpy.diag(S)
    dTest = numpy.dot(dU, numpy.dot(S, VT)) +\
            numpy.dot(U, numpy.dot(numpy.diag(dS), VT)) +\
            numpy.dot(U, numpy.dot(S, dV.T))

    norm_check_diffSVD = linalg.norm(dTest - dY, 'fro')/linalg.norm(dY, 'fro')
    if norm_check_diffSVD < 1.0e-10:
        print('SVD-diff worked!')
        print('norm_check_diffSVD=', norm_check_diffSVD)
    else:
        print('SVD-diff did not work!')
        print('norm_check_diffSVD=', norm_check_diffSVD)

    #--------------------------------------------------------------------------
    #TEST FOR TRUNCATED SVD!!!
    #--------------------------------------------------------------------------
    Y1  = random.rand(10000, 30)
    dY1 = random.rand(10000, 30)
    Y2  = random.rand(30, 1000)
    dY2 = random.rand(30, 1000)
    #--------------------------------------------------------------------------
    #derivative of matrix function
    #  Y(t) =  (Y1+tdY1)*(Y2+tDY2)
    #--------------------------------------------------------------------------
    
    Y  = numpy.dot(Y1, Y2)
    dY = numpy.dot(dY1, Y2) +  numpy.dot(Y1, dY2)
    print('rank Y, rank dY:', numpy.linalg.matrix_rank(Y), ',', numpy.linalg.matrix_rank(dY))
    
    U, S, VT = scipy.linalg.svd(Y,\
                                full_matrices=False,\
                                compute_uv=True,\
                                overwrite_a=False)
    # remark: scipy returns rectangular U but quadratic V
 
    # TRUNCATE
    tau = 1.0e-5
    St = S[S>tau]
    #St = S[0:30]
    p = St.shape[0]
    print('p =', p)
    # keep only the first p columns
    Ut = numpy.matrix(U[:,0:p])
    Vt = numpy.matrix(VT.T[:,0:p])
    
    # diff-algorithm needs full orthogonal V
    dU_bot, A, dU, Alpha, dS, dV = Stiefel_diff_SVD(Y, dY, Ut, St, VT.T)
    # does it match the derivative Y(t)?
    St = numpy.diag(St)
    # the output of "Stiefel_diff_SVD" is already the truncated matrices
    dTest = numpy.dot(dU, numpy.dot(St, Vt.T)) +\
            numpy.dot(Ut, numpy.dot(numpy.diag(dS), Vt.T)) +\
            numpy.dot(Ut, numpy.dot(St, dV.T))

    norm_check_diffSVD = linalg.norm(dTest - dY, 'fro')/linalg.norm(dY, 'fro')
    if norm_check_diffSVD < 1.0e-10:
        print('SVD-diff worked!')
        print('truncated norm_check_diffSVD=', norm_check_diffSVD)
    else:
        print('SVD-diff did not work!')
        print('truncated norm_check_diffSVD=', norm_check_diffSVD)
    #check sing val derivatives:
    # finite differences:
    h = 1.0e-6
    Yh  = numpy.dot(Y1+h*dY1, Y2+h*dY2)   
    Uh, Sh, VTh = scipy.linalg.svd(Yh,\
                                full_matrices=False,\
                                compute_uv=True,\
                                overwrite_a=False)
    Yhm  = numpy.dot(Y1-h*dY1, Y2-h*dY2)   
    Uhm, Shm, VThm = scipy.linalg.svd(Yhm,\
                                full_matrices=False,\
                                compute_uv=True,\
                                overwrite_a=False)
    S_FD = (Sh[0:p]-Shm[0:p])/(2*h)
    print('compared to finite differences', scipy.linalg.norm(S_FD-dS))
    