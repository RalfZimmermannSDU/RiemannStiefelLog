#------------------------------------------------------------------------------
# Auxiliary functions in relation to computing Riemannian Barycenters
#------------------------------------------------------------------------------
import numpy as np
import scipy 
import sys
from   scipy import linalg
import time 
from numpy import random
import matplotlib.pylab as plt

sys.path.append('../../../Stiefel_log_general_metric/SciPy/')

import Stiefel_retractions as STR
import Stiefel_Aux as StAux
import Stiefel_Exp_Log as StEL

def objective_RBC(U,Us,ws,mode = 1):
    n,p = U.shape
    m = Us.shape[0]
    f = 0
    for k in range(m):
        if mode == 1: # Riemannian logarithm
            f += ws[k,0] * StEL.distStiefel(U,Us[k,:,:],-0.5) ** 2
        elif mode == 2:
            f += ws[k,0] * np.linalg.norm(STR.Stiefel_PF_inv_ret(U,Us[k,:,:]),'fro')**2
        elif mode == 3:
            f += ws[k,0] * np.linalg.norm(STR.Stiefel_PL_inv_ret(U,Us[k,:,:]),'fro')**2
        elif mode == 4:
            f += ws[k,0] * np.linalg.norm(STR.Stiefel_PL_inv_ret(U,Us[k,:,:],2),'fro')**2
        else:
            f = 0
            
    
    return f
    

def gradient_RBC(U,Us,ws,mode = 1):
    n,p = U.shape
    m = Us.shape[0]
    grad = np.zeros((n,p))
    for k in range(m):
        if mode == 1: # Riemannian logarithm
            L,c = StEL.Stiefel_Log(U,Us[k,:,:],1e-11,-0.5)
            grad += -1*ws[k,0] * L
        elif mode == 2: # Use PF
            L = STR.Stiefel_PF_inv_ret(U,Us[k,:,:])
            grad += -1*ws[k,0] * L
        elif mode == 3: # Use PL
            L = STR.Stiefel_PL_inv_ret(U,Us[k,:,:])
            grad += -1*ws[k,0] * L
        elif mode == 4: # Use PL Cay
            L = STR.Stiefel_PL_inv_ret(U,Us[k,:,:],2)
            grad += -1*ws[k,0] * L
        else: # Implement inverse retractions
            grad = 0
    
    return grad

