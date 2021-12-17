#---------------------------------------------------------------------------
#
# This file implements a numerical validation check that is used in
# the script
#
# script_Hermite_St_Interp_SISC_test_eq25_Section_S5.py
#
# that is to be found in
#
# Applications/SciPy/Stiefel_interp_applications/
#---------------------------------------------------------------------------



import scipy
import numpy as np
import sys

sys.path.append('../../../Stiefel_log_general_metric/SciPy/')

import Stiefel_Exp_Log          as StEL



#------------------------------------------------------------------------------
# check derivative of Stiefel curve
# * compute geodesic gamma0 from U0 to U1
# * compute geodesic gamma1 from U1 to U0
# * check if derivative  gamma1'(1) matches -gamma0'(0)
#------------------------------------------------------------------------------
def test_diff_Stiefel_curve(U0, U1, do_checks, metric_alpha=0.0):
    # compute gamma0: 

    # step 1: get tangent vector gamma0'(0)
    Delta01, iter_count =  StEL.Stiefel_Log(U0, U1, 1.0e-13, metric_alpha)

    # step 2: from U1 to U0
    Delta10, iter_count =  StEL.Stiefel_Log(U1, U0, 1.0e-13, metric_alpha)
    
    # step 3: get tangent vector gamma1'(1)
        # get dimensions
    n,p = U1.shape
    A = np.dot(U1.transpose(),Delta10)  # horizontal component
    K = Delta10-np.dot(U1,A)            # normal component
    
    # qr of normal component
    Q, R = scipy.linalg.qr(K, overwrite_a=True,\
                              lwork=None,\
                              mode='economic',\
                              pivoting=False,\
                              check_finite=True)
                                        
    # matrix exponential
    # metric_factor
    v = 1.0/(metric_alpha +1.0)
    
    upper = np.concatenate((v*A, -R.transpose()), axis=1)
    lower = np.concatenate((R, np.zeros((p,p), dtype=R.dtype)), axis=1)
    L     = np.concatenate((upper, lower), axis=0)
    MNe   = scipy.linalg.expm(L)

    if abs(metric_alpha) < 1.0e-12:
        # canonical metric
        # perform gamma1'(1) = (U1, Q)*MNe*L*(I,0)^T
        H = np.dot(MNe, L[:,0:p]) 
        ddt_gamma1 = np.dot(U1,H[0:p,0:p]) +  np.dot(Q, H[p:2*p,0:p])
    else:
        # alpha metric
        # metric_factor
        mu   = metric_alpha/(metric_alpha +1.0)
        expA = scipy.linalg.expm(mu*A)
        # perform gamma1'(1) = (U1, Q)*MNe*((A;R)*expA)
        L = np.concatenate((A, R), axis=0)
        L = np.dot(L, expA)
        H = np.dot(MNe, L[:,0:p]) 
        ddt_gamma1 = np.dot(U1,H[0:p,0:p]) +  np.dot(Q, H[p:2*p,0:p])

    
    if do_checks:
        checknorm1 = np.linalg.norm(ddt_gamma1 + Delta01, 'fro')/np.linalg.norm(Delta01, 'fro')
        print("Is the tangent vector of Exp_U1(t*Delta10) at t=1")
        print("the same as minus the tangent vector of Exp_U0(t*Delta01) at t=0?")
        print("The two quantities match up to an error of ", checknorm1)   
    
        S = np.dot(ddt_gamma1.transpose(),U0)
        checknorm2 = np.linalg.norm(S+S.T, 'fro')
        print("Is d/dt|t=1 Exp_U1(t*Delta10) in T_U0St(n,p)? ", checknorm2)
        
        # compute endpoint
        if abs(metric_alpha) < 1.0e-10:
            # canonical
            U0_rec = np.dot(U1,MNe[0:p,0:p]) +  np.dot(Q, MNe[p:2*p,0:p])
        else:
            # alpha-metric
            MNe = np.dot(MNe[:,0:p], expA)
            U0_rec = np.dot(U1,MNe[0:p,0:p]) +  np.dot(Q, MNe[p:2*p,0:p])
        checknorm3 = np.linalg.norm(U0-U0_rec, 'fro')
        print("Is Exp_U1(t*Delta10) crossing U0 at t=1? ",checknorm3)
    return Delta01, ddt_gamma1
#------------------------------------------------------------------------------
 