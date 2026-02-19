import numpy as np
import scipy
from scipy import linalg
import time

import Stiefel_Exp_Log
import Stiefel_Aux        as StAux

# set dimensions
n = 1000
p = 400
    


# set number of random experiments
runs = 1
dist = 1.2*np.pi
tau =  1.0e-11

#initialize

time_array  = np.zeros((5,))

#for the Euclidean metric: alpha = -0.5
metric_alpha = -0.5 
for j in range(runs):
    #----------------------------------------------------------------------
    #create random stiefel data
    U0, U1, Delta = Stiefel_Exp_Log.create_random_Stiefel_data(n, p, dist, metric_alpha)
    #----------------------------------------------------------------------
    
    M = (-1)*np.dot(U0.T,U1)
    # solve MX + XM = -2*eye(p)
    t_start = time.time()
    X = scipy.linalg.solve_sylvester(M, M.T, (-2)*np.eye(p,p))
    t_end = time.time()
    time_array[0] =  time_array[0] + t_end-t_start
    
    # costs of SVD
    t_start = time.time()    
    D, S, R = scipy.linalg.svd(M,\
                               full_matrices=True,\
                               compute_uv=True,\
                               overwrite_a=False)
    t_end = time.time()
    time_array[1] =  time_array[1] + t_end-t_start
    
    
    # costs of matrix log
    L= np.dot(D, R.T)
    t_start = time.time() 
    
    [LV, flag_negval] = StAux.SchurLog(L)
    t_end = time.time()
    time_array[2] =  time_array[2] + t_end-t_start
    
    
    t_start = time.time() 
    Test   = Stiefel_Exp_Log.Stiefel_Pf_Ret(U0, Delta)
    Delta2 = Stiefel_Exp_Log.Stiefel_Pf_invRet(U0, Test)
    t_end = time.time()
    
    
        
        
    print('here')
    print(np.allclose(np.dot(Test.T,Test), np.eye(p)))
    print(np.linalg.norm((Delta-Delta2)))
    print(time_array)
    print(time_array[0] > time_array[1]+time_array[2])
