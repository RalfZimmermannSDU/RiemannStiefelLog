#------------------------------------------------------------------------------
#@author: Ralf Zimmermann, IMADA, SDU Odense
# zimmermann@imada.sdu.sdk
#
# Algorihtms associated with the publication
#
# R. Zimmermann, K. H\"uper.
# "Computing the Riemannian logarithm on the Stiefel manifold: 
#  metrics, methods and performance", arXiv:2103.12046, March 2022
#
#
#
# This file implements the following methods:
#
# * Stiefel_Exp(U0, Delta, metric_alpha):
#       - compute U1 = Exp^{St}_U0(Delta) for the alpha-metric
#
# * Stiefel_Log(U0, U1, tau, metric_alpha)
#       - wrapper for the various Stiefel log algs, depending 
#         on the metric parameter alpha
#
# * Stiefel_Log_alg(U0, U1, tau, do_Procrustes=0, do_Cayley=0, do_Sylvester=1):
#       - compute Delta = Log^{St}_U0(U1), up to an accuracy of tau
#         via algebraic Stiefel log for the canonical metric
#
# * Stiefel_Log_p_Shooting_uni(U0, U1, unit_int, tau, metric_alpha):
#       - compute Delta = Log^{St}_U0(U1), up to an accuracy of tau
#         via a shooting Stiefel log for the alpha-metric
#
# * distStiefel(U1, U2, metric_alpha):
#       - compute dist(U1,U2) = || Log_U1(U2)||
#
# * create_random_Stiefel_data(n, p, dist, metric_alpha):
#       - create (pseudo-)random Stiefel data: for testing purposes
#
#------------------------------------------------------------------------------

import numpy
import scipy
from scipy import linalg
from numpy import random

#prog_path = '/home/zimmermann/Documents/ScientificAndSVN/RZ_SDU/Papers_IMADA/Manifold_Ops/Manifold_Ops_Python/'
#sys.path.append(prog_path + 'Stiefel_Aux/')
import Stiefel_Aux        as StAux

###############################################################################
# FUNCTION DECLARATION
###############################################################################


#------------------------------------------------------------------------------
# Exponential map on Stiefel manifold
#
# Following 
#
# A. Edelman, T. A. Arias, and S. T. Smith. 
# "The geometry of algorithms with orthogonality constraints." 
# SIAM Journal on Matrix Analysis and
# Applications, 20(2):303-353, 1998.
#
# extended to family of alpha-metrics according to 
#
# R. Zimmermann, K. H\"uper.
# "Computing the Riemannian logarithm on the Stiefel manifold: 
#  metrics, methods and performance", arXiv:2103.12046, March 2022
# 
# Input arguments      
#          U0    : base point on St(n,p)
#          Delta : tangent vector in T_U0 St(n,p)
#   metric_alpha : metric parameter
# Output arguments
#          U1    : Exp^{St}_U0(Delta),
#------------------------------------------------------------------------------
def Stiefel_Exp(U0, Delta, metric_alpha=0.0):
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
    if scipy.allclose(0.0, metric_alpha):
        # special case: alpha=0: canonical metric
        # build block matrix, matrix exponential
        upper = scipy.concatenate((A, -Re.transpose()), axis=1)
        lower = scipy.concatenate((Re, scipy.zeros((p,p), dtype=Re.dtype)), axis=1)
        L     = scipy.concatenate((upper, lower), axis=0)
        MNe   = scipy.linalg.expm(L)
        MNe   = MNe[:,0:p]
    elif (abs(metric_alpha+1.0)> 1.0e-13):
        # case: alpha-metric 
        x  = 1.0/(metric_alpha +1.0)
        y  = (metric_alpha)/(metric_alpha +1)
        # build block matrix, evaluate matrix exponential
        upper = scipy.concatenate((x*A, -Re.transpose()), axis=1)
        lower = scipy.concatenate((Re, scipy.zeros((p,p), dtype=Re.dtype)), axis=1)
        L     = scipy.concatenate((upper, lower), axis=0)
        MNe   = scipy.linalg.expm(L)
        expA  = scipy.linalg.expm(y*A)
        MNe   = scipy.dot(MNe[:,0:p]       , expA)
    else:
        print('Error in  Stiefel_Exp: wrong metric. Choose alpha != -1.')
        print('Returning U1=U0')
        MNe   = scipy.eye(2*p,p)
        
    # perform U1 = U0*M + Q*N
    U1 = scipy.dot(U0,MNe[0:p,0:p]) +  scipy.dot(QE, MNe[p:2*p,0:p])
    return U1
#------------------------------------------------------------------------------




#------------------------------------------------------------------------------
# Wrapper function for Riemannian Stiefel logarithm
# 
# Compute Stiefel log w.r.t. to the alpha metric
# By default, use
#  *algebraic Stiefel log for the canonical metric
#  *shooting method for any other alpha metric
#
# Input arguments      
#      U0, U1 : points on St(n,p)
#         tau : convergence threshold
#metric_alpha : metric parameter
# Output arguments
#   Delta : Log^{St}_U0(U1), 
#           i.e. tangent vector such that Exp^St_U0(Delta) = U1
#  conv_hist : convergence history
#------------------------------------------------------------------------------
def Stiefel_Log(U0, U1, tau, metric_alpha=0.0):
#------------------------------------------------------------------------------

    if abs(metric_alpha) < 1.0e-13:
        # canonical metric: use algebraic Stiefel log
        print("Use algebraic Stiefel log, metric alpha = ",metric_alpha)
        Delta, conv = Stiefel_Log_alg(U0, U1, tau)
    elif abs(metric_alpha + 1.0) > 1.0e-8:
        print("Use shooting Stiefel log, metric alpha = ",metric_alpha )
        unit_int = scipy.linspace(0.0,1.0,4)
        Delta, conv = Stiefel_Log_p_Shooting_uni(U0,\
                                                 U1,\
                                                 unit_int,\
                                                 tau,\
                                                 metric_alpha)
    else:
        print('Wrong metric parameter in <Stiefel_Log>')
        Delta = 0
        conv = []
    return Delta, conv
#------------------------------------------------------------------------------






#------------------------------------------------------------------------------
# Riemannian Stiefel logarithm
# 
# following
#
# R. Zimmermann, K. H\"uper.
# "Computing the Riemannian logarithm on the Stiefel manifold: 
#  metrics, methods and performance", arXiv:2103.12046, March 2022
#
# Input arguments      
#  U0, U1 : points on St(n,p)
#     tau : convergence threshold
#
# Meta-parameters:
# do_Procrustes = 0/1: Do Procrustes preprocessing?
# do_Cayley     = 0/1: use Cayley trafo to approximate the matrix exp.
# do_Sylvester  = 0/1: solve Sylvester euqation for improved converegence rate
# Output arguments
#   Delta : Log^{St}_U0(U1), 
#           i.e. tangent vector such that Exp^St_U0(Delta) = U1
#       k : iteration count upon convergence
#------------------------------------------------------------------------------
def Stiefel_Log_alg(U0, U1, tau, do_Procrustes=0, do_Cayley=0, do_Sylvester=1):
#------------------------------------------------------------------------------
    # get dimensions
    n,p = U0.shape
    # check_det     : 0/1, check if initial V0 is in SO(2p)
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
    V, R = scipy.linalg.qr(MN, overwrite_a=True,\
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
    
    # check if V \in SO(2p)
    if check_det:
        # ensure that V \in SO(n)                                       
        DetV = scipy.linalg.det(V)
        if DetV < 0:
            # flip sign of one column
            V[:,p] = (-1)*V[:,p]

    # initialize convergence history      
    conv_hist = []                                                     
    # step 4: FOR-Loop
    for k in range(1000):
        # step 5
        # home-brew logm-alg. tailored for orthogonal matrices from Stiefel_Aux
        # as an alternative, use: LV = linalg.logm(V)
        LV = StAux.SchurLog(V)
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
        if do_Sylvester:
            # indices of diagonal p by p matrix
            diag_pp = scipy.diag_indices(p)
            # set up symmetric Sylvester problem
            # compute (1.0/12.0)*B*B' - 0.5*eye(p)
            # Caution: the block LV(p+1:2*p) contains -B' !
            #          need to correct for the correct sign
            Msym =(-1.0/12.0)*scipy.dot(LV[p:2*p, 0:p], LV[0:p, p:2*p])
            Msym[diag_pp] = Msym[diag_pp] - 0.5
            # solve Sylvester equation
            Csylv = scipy.linalg.solve_sylvester(Msym, Msym, C)
            # make Csylv exactly skew
            # for both cases do_Sylvester = 0,1,
            # eventually exp(-C) is formed =>return -Csylv here.
            C = -0.5*(Csylv-Csylv.T)

        # exponential of updated block
        if do_Cayley:
            # Cayley approx
            Phi = StAux.Cayley(-C)
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
# Shooting method for computing the Stiefel logarithm
# for all alpha-metrics
# Calculations restricted to pxp-matrices in the loop
#
# 
# following
#
# R. Zimmermann, K. H\"uper.
# "Computing the Riemannian logarithm on the Stiefel manifold: 
#  metrics, methods and performance", arXiv:2103.12046, March 2022
#
# Input arguments      
#      U0, U1 : points on St(n,p)
#    unit_int : discrete representation of unit interval [0,1]
#         tau : convergence threshold
#metric_alpha : matric parameter
#               use alpha =-1/2 for 'euclid' 
#               use alpha =   0 for 'canonical'
#
# Output arguments
#   Delta : Log^{St}_U0(U1), 
#           i.e. tangent vector such that Exp^St_U0(Delta) = U1
#       k : iteration count upon convergence
#@author: Ralf Zimmermann, IMADA, SDU Odense
# 
#------------------------------------------------------------------------------
def Stiefel_Log_p_Shooting_uni(U0, U1, unit_int, tau, metric_alpha):
#------------------------------------------------------------------------------
    n,p = U0.shape

    # controlling parameters
    max_iter = 200

    tsteps = len(unit_int)

    # step 1: compute the fixed coordinates U, Q
    M0      = scipy.dot(U0.T,U1)

    # thin qr of normal component of U1
    Q, R0 = scipy.linalg.qr(U1 - scipy.dot(U0,M0),\
                            overwrite_a=True,\
                            lwork=None,\
                            mode='economic',\
                            pivoting=False,\
                            check_finite=True)
    # initial gap vector W = U1-U0 needs not be formed
    # W = U0*M0 + Q0*R0 - U0 = U0(M0-I) + Q*R0

    # compute norm of W, this matches norm(U0*(M0-I) + Q0*R0, 'fro')
    n_M0I = scipy.linalg.norm(M0-scipy.eye(p),'fro')
    n_R0  = scipy.linalg.norm(R0,'fro') 
    n_w   = scipy.sqrt(n_M0I**2 + n_R0**2)

    # compute initial shooting vector:
    # project gap U1-U0 onto T_U0 St(n,p)
    A = StAux.A2skew(M0)
    R = R0
    #now: Delta = U0*A + Q0*R, no need to form explicitly

    # scale Delta to norm of W
    n_d = scipy.sqrt(scipy.dot(A.flatten(), A.flatten()) + n_R0**2)
    A = (n_w/n_d)*A
    R = (n_w/n_d)*R0

    # initialize array of "geodesic p-factors" at each t in t_int
    GeoM = scipy.zeros((tsteps, p,p))
    GeoN = scipy.zeros((tsteps, p,p))
    GeoM[0,:,:] = scipy.eye(p)

    # make sure that the iterations start
    j = 0
    conv_hist = []
  
    while (n_w > tau) and (j<max_iter):
        j = j+1
    
        # evaluate geodesic factors at the discrete steps
        # for the current Delta = U A + Q R
        if tsteps > 2:
            # in this case, it is more efficient to 
            # first compute the EVD of [[xA, -R'][R, 0]] and then 
            # the matrix exp for all t-steps.
            if abs(metric_alpha - 0.0) < 1.0e-13:                   # canonical metric
                A_pre = A
            elif abs(metric_alpha+1.0) > 1.0e-13:
                x = 1.0/(metric_alpha +1)
                A_pre = x*A
            else:
                A_pre = A
            
            upper = scipy.concatenate((A_pre, -R.transpose()), axis=1)
            lower = scipy.concatenate((R, scipy.zeros((p,p), dtype=R.dtype)), axis=1)
            L     = scipy.concatenate((upper, lower), axis=0)                 
            Evals, Evecs = scipy.linalg.eig(L)               
            # eigenvalues are on complex axis
            evals = scipy.imag(Evals)
        
            for k in range(1,tsteps):
                GeoM[k,:,:],GeoN[k,:,:] = StAux.Exp4Geo_pre(unit_int[k],\
                                                            A_pre,\
                                                            Evecs,\
                                                            evals,\
                                                            metric_alpha)
        else:            # i.e., here tsteps <= 2, which means t_steps==2 
            GeoM[1,:,:],GeoN[1,:,:] = StAux.Exp4Geo(A, R, metric_alpha)
        # end: if t_steps ...
        # compute new gap vector 
        # W = Geo(1) - U1 
        #   = U0*Mend + Q0*Nend - U1 = U0*(Mend-M0) + Q0*(Nend-R0)
        # need only the updated A and R factors
        A_up = GeoM[-1,:,:]-M0
        R_up = GeoN[-1,:,:]-R0
    
        # compute norm of W
        # this matches norm(U0*M + Q0*N - U1, 'fro')    
        n_w = scipy.sqrt(scipy.linalg.norm(A_up,'fro')**2\
                         + scipy.linalg.norm(R_up,'fro')**2)
        conv_hist.append(n_w)
    
        # paralelle translate W to T_U0St(n,p):
        #   project gap onto T_(Geo(end))
        #   and then along the geodesic onto T_(Geo(0))
        for k in range(tsteps):
            A_up, R_up = StAux.Stiefel_approx_parallel_trans_p(GeoM[tsteps-k-1,:,:],\
                                                         GeoN[tsteps-k-1,:,:],\
                                                         A_up,\
                                                         R_up,\
                                                         n_w)
    
        # update Delta= Delta - d*W, using the p-factors only
        A = A - A_up
        R = R - R_up

    # form Delta
    Delta = scipy.dot(U0,A)+scipy.dot(Q,R)

    if j < max_iter:
        print('p-Shooting unified method converged in ', str(j), ' iterations')
    else:
        print('p-Shooting unified method did not converge')
    
    return Delta, conv_hist
#------------------------------------------------------------------------------
    

#------------------------------------------------------------------------------
# Riemmannian distance on St(n,p)
#
# input arguments
#       U1, U2 = points on St(n,p)
# metric_alpha = metric parameter
#
# output arguments
#   dist(U1,U2) = || Log_U1(U2)|| for the chosen metric
#
#------------------------------------------------------------------------------
def distStiefel(U1, U2, metric_alpha=0.0):
#------------------------------------------------------------------------------
    tau = 1.0e-11
    Delta, conv = Stiefel_Log(U1, U2, tau, metric_alpha)      
    dist = scipy.sqrt(StAux.alphaMetric(Delta, Delta, U1, metric_alpha))
    return dist
#------------------------------------------------------------------------------




#------------------------------------------------------------------------------
# create a random data set 
# U0, U1 on St(n,p),
#  Delta on T_U St(n,p) with norm 'dist',
# which is also the Riemannian distance dist(U0,U1)
#
# input arguments
#        (n,p) = dimension of the Stiefel matrices
#        dist  = Riemannian distance between the points U0,U1
#                that are to be created
# metric_alpha = metric parameter
#------------------------------------------------------------------------------
def create_random_Stiefel_data(n, p, dist,metric_alpha=0.0):
#------------------------------------------------------------------------------
    #create random Stiefel matrix:
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
    Delta = scipy.dot(U0,A)+ T-scipy.dot(U0,scipy.dot(U0.transpose(),T))
    #normalize Delta w.r.t. the canonical metric
    norm_Delta = scipy.sqrt(StAux.alphaMetric(Delta, Delta, U0,metric_alpha))
    Delta = (dist/norm_Delta)*Delta
    # 'project' Delta onto St(n,p) via the Stiefel exponential
    U1 = Stiefel_Exp(U0, Delta,metric_alpha)
    return U0, U1, Delta
#------------------------------------------------------------------------------




































#******************************************************************************
#  ||
#  ||     Down here: testing 
# \ /
# \/
#******************************************************************************

do_tests = 0
if do_tests:
    
    # set dimensions
    n = 200
    p = 90
    
    #for the Euclidean metric: alpha = -0.5
    #for the Canonical metric: alpha = 0.0
    metric_alpha = -0.0

    # set number of random experiments
    runs = 1
    dist = 1.2*scipy.pi
    tau =  1.0e-11

    #initialize
    iters_array = scipy.zeros((5,))
    time_array  = scipy.zeros((5,))
    is_equal    = scipy.zeros((5,))
    
    for j in range(runs):
        #----------------------------------------------------------------------
        #create random stiefel data
        U0, U1, Delta = create_random_Stiefel_data(n, p, dist, metric_alpha)
        #----------------------------------------------------------------------
        print('dist', distStiefel(U0, U1, metric_alpha))

        # basic exp log test:
        # compare three methods to compute the exponential
        A = scipy.dot(U0.T,Delta)
        A = StAux.A2skew(A)
        Q, R = scipy.linalg.qr((Delta-scipy.dot(U0,A)), overwrite_a=True,\
                               lwork=None,\
                               mode='economic',\
                               pivoting=False,\
                               check_finite=True) 

        if abs(metric_alpha - 0.) < 1.0e-13:                      # canonical metric
            A_pre = A
        elif abs(metric_alpha+1) > 1.0e-13:
            x  = 1.0/(metric_alpha +1)
            A_pre = x*A
        else:
            A_pre = A

        upper = scipy.concatenate((A_pre, -R.transpose()), axis=1)
        lower = scipy.concatenate((R, scipy.zeros((p,p), dtype=R.dtype)), axis=1)
        L     = scipy.concatenate((upper, lower), axis=0)                 
        Evals, Evecs = scipy.linalg.eig(L)         
        # eigenvalues are on complex axis
        evals = scipy.imag(Evals)
        
        M, N = StAux.Exp4Geo_pre(1.0, A_pre, Evecs, evals, metric_alpha)
        
        EXP1 = scipy.dot(U0,M) + scipy.dot(Q,N)
        EXP2 = Stiefel_Exp(U0, Delta,metric_alpha)
        
        M, N = StAux.Exp4Geo(A, R, metric_alpha)
        EXP3 = scipy.dot(U0,M) + scipy.dot(Q,N)
        
        print('NORM TEST1:', scipy.linalg.norm(U1-EXP2, 1))
        print('NORM TEST2:', scipy.linalg.norm(U1-EXP1, 1))
        print('NORM TEST3:', scipy.linalg.norm(U1-EXP3, 1))
# End: if do_tests



















