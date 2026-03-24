#------------------------------------------------------------------------------
# Compute the Riemannian center of mass of a geodesic triangle
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
import barycenters_aux as BAux

np.random.seed(345894)
# Set dimensions
n = 200
p = 30

alpha  = -0.5

# Optimization parameters
Nmax = 1000
tau = 1e-7
delta = 0.5
tol = 10e-10

A = np.random.rand(n,p)
U0,R0 = np.linalg.qr(A,mode='reduced')

Us = np.zeros((3,n,p))
Xis = np.zeros((2,n,p))
ws = np.ones((3,1))*1/3

Us[0,:,:] = U0
for i in range(1,3):
    A = np.random.rand(p,p) 
    A = 0.5 * (A.T - A) # Now A is p x p skew
    A = A / np.linalg.norm(A,'fro')

    B = np.random.rand(n,p) 
    B = B / np.linalg.norm(B,'fro')
    B = np.zeros([n,p])

    Xi = U0 @ A + (B - U0 @ (U0.T @ B))
    Xis[i-1,:,:] = Xi / np.linalg.norm(Xi) * 0.8 * np.pi

    Us[i,:,:] = StEL.Stiefel_Exp(U0,Xis[i-1,:,:],alpha)

# Angle
theta = np.acos(np.abs(np.trace(Xis[0,:,:].T @ Xis[1,:,:]))/(np.linalg.norm(Xis[0,:,:],'fro')*np.linalg.norm(Xis[1,:,:],'fro')))
print(theta)
# Checks
# print(StEL.distStiefel(U0, Us[1,:,:], alpha))
# print(StEL.distStiefel(U0, Us[2,:,:], alpha))
# print("Intended dist: ", str(0.8*np.pi))
# print(StEL.distStiefel(Us[1,:,:], Us[2,:,:], alpha))


U0 = np.copy(Us[0,:,:]) # Initial guess

modes = [1,2,3,4]

GradNorm = np.zeros((len(modes),Nmax))
TimeStamp = np.zeros((len(modes),Nmax))
fVals = np.zeros((len(modes),Nmax))
N_iter = np.zeros((len(modes),1))
Umus = np.zeros((len(modes),n,p))
#modes = [2,3,4]

for mode in modes:
    U = np.copy(U0)
    # k = 0
    # # Record at time 0
    # grad = BAux.gradient_RBC(U,Us,ws,mode)
    # GradNorm[mode-1,k] = np.linalg.norm(grad,'fro')
    # fVals[mode-1,k] = BAux.objective_RBC(U,Us,ws,mode)
    
    k = 0
    fVals[mode-1,k] = BAux.objective_RBC(U,Us,ws,mode)
    while k <= Nmax-2:

        if mode == 1:
        # Compute gradient 
            t1 = time.time()
            grad = BAux.gradient_RBC(U,Us,ws,mode)
            U = StEL.Stiefel_Exp(U,-delta*grad)
            t2 = time.time()
        elif mode == 2: # PF
            t1 = time.time()
            grad = BAux.gradient_RBC(U,Us,ws,mode)
            U = STR.Stiefel_PF_ret(U,-delta*grad)
            t2 = time.time()
        elif mode == 3: # PL 
            t1 = time.time()
            grad = BAux.gradient_RBC(U,Us,ws,mode)
            U = STR.Stiefel_PL_ret(U,-delta*grad)
            t2 = time.time()
        elif mode == 4: # PL Cayley
            t1 = time.time()
            grad = BAux.gradient_RBC(U,Us,ws,mode)
            U = STR.Stiefel_PL_ret(U,-delta*grad,mode = 2)
            t2 = time.time()
        else:
            grad = BAux.gradient_RBC(U,Us,ws,1)
            U = U0

        GradNorm[mode-1,k] = np.linalg.norm(grad,'fro')
        # Check if converged
        if GradNorm[mode-1,k] < tol:
            # We terminate at iteration k - 1
            Umus[mode - 1,:,:] = U
            break
        else:
            if k % 10 == 0:
                print('Mode = ',mode,' Iteration k = ',k)
            k = k + 1
            # Record time 
            TimeStamp[mode-1,k] = TimeStamp[mode-1,k-1] + (t2 - t1)  
            fVals[mode-1,k] = BAux.objective_RBC(U,Us,ws,mode)
        
            


    N_iter[mode-1,0] = k
    print("***  Ustar minus U1 and U2  ***")
    print("Ustar,U1",StEL.distStiefel(U, Us[0,:,:], alpha))
    print("Ustar,U1",StEL.distStiefel(U, Us[1,:,:], alpha))
    print("Ustar,U1",StEL.distStiefel(U, Us[2,:,:], alpha))
    print("Sum     ",StEL.distStiefel(U, Us[0,:,:], alpha)+StEL.distStiefel(U, Us[1,:,:], alpha)+StEL.distStiefel(U, Us[2,:,:], alpha))
    print("Time    ",TimeStamp[mode-1,k])
    print("N_iter  ",N_iter[mode-1,0])
    print("G. norm ",GradNorm[mode-1,k])
    print("*** ** ** ** ** ** ** ** ** ***")

print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
for i in range(1,4):
    print("||Umu-Umu_Rie|| mode=",i+1," is ", np.linalg.norm(Umus[0,:,:] - Umus[i,:,:],'fro' ))
    #print("||Umu-Umu_Rie|| mode=",i+1," is ", StEL.distStiefel(Umus[0,:,:], Umus[i,:,:], alpha))
print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")

            
print(N_iter)
# Checks
# print(StEL.distStiefel(U, Us[1,:,:], alpha))
# print(StEL.distStiefel(U, Us[2,:,:], alpha))


# Export data
np.save("GradNorm",GradNorm)
np.save("N_iter",N_iter)
np.save("TimeStamp",TimeStamp)
np.save("fVals",fVals)