

#------------------------------------------------------------------------------
# Compare the cost of evolving a retraction as time increases. 
# In such a setting, precomputing several quantities is possible, so that 
# we only have to compute the true t-dependent part of the retractions
#
# We consider:
# Polar factor retraction 
# Polar light retraction w. and wout. Cayley 
# Quasi geodesic w. and wout. Cayley
#
# Pipeline: 
# Obtain tangent vector from U0 and U1 -> compute points as t = 0,...,1 for d
# equidistant points. Timing of the preprocessing step is timed separately.
# Timing of the precomputing step is timed separately.
# 
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
# Dimensions 

np.random.seed(345345)



def run_timings(n,p):
    A = np.random.rand(n,p)
    U0,R0 = np.linalg.qr(A,mode='reduced')

    # Generate a tangent vector at U0
    A = np.random.rand(p,p) 
    A = 0.5 * (A.T - A) # Now A is p x p skew
    A = A / np.linalg.norm(A,'fro')

    B = np.random.rand(n,p) 
    B = B / np.linalg.norm(B,'fro')
    #B = np.zeros([n,p])

    Xi = U0 @ A + (B - U0 @ (U0.T @ B))

    Q,R = np.linalg.qr(Xi - U0 @ (U0.T @ Xi),mode = 'reduced')
    ART = np.concatenate((A,-R.T),axis =1)
    RZ  = np.concatenate((R, np.zeros([p,p])),axis=1)
    S = np.concatenate((ART, RZ),axis=0)
    UQ = np.concatenate([U0, Q],axis=1)

    U1 = UQ @ scipy.linalg.expm(S)
    U1 = U1[:,0:p]

    #print(np.linalg.norm(U1.T @ U1 - np.eye(p)))

    # Time interval
    I = np.linspace(0,1,num = 11)

    # Compute geodesic for reference
    # T,W = scipy.linalg.schur(S,output = 'real')
    # #T,W = scipy.linalg.eig(S)
    # UQW = UQ @ W


    Data = np.zeros([6,n,p])
    Time_data = np.zeros([5,3])
    # # ti = time.time()
    # # Schur is not making the computations faster. 
    # for t in I:
    #     Ut = UQW @ (scipy.linalg.expm(t*T) @ W.T)
    #     #Ut = UQ @ scipy.linalg.expm(t*S)
    #     Data[0,:,:] = Ut[:,0:p]
        
    # tend = time.time()
    # print("Time ", tend - ti)
    # ti = time.time()
    # for t in I:
    #     #Ut = UQW @ (scipy.linalg.expm(t*T) @ W.T)
    #     Ut = UQ @ scipy.linalg.expm(t*S)
    #     Data[0,:,:] = Ut[:,0:p]
        
    # tend = time.time()
    # print("Time ", tend - ti)

    #print(np.linalg.norm(Ut[:,0:p] - U1))

    print("*** Running Polar factor retraction ***")

    # Step 1: Obtain Xi
    t0 = time.time()
    Xi = STR.Stiefel_PF_inv_ret(U0,U1)

    # Check that we recover U1
    # print(np.linalg.norm(U1 - STR.Stiefel_PF_ret(U0,Xi)))

    # For the computation
    t1 = time.time()
    S = np.linalg.qr(Xi, mode='r')
    M, Sing, VT = scipy.linalg.svd(S,\
                                full_matrices=True,\
                                compute_uv=True,\
                                overwrite_a=True)
    t2 = time.time()
    for t in I:
        Sing_t = np.sqrt(1/(t**2 * Sing**2 + 1))
        STS_t = (VT.T*Sing_t) @ VT
        Ut = (U0 + t * Xi) @ STS_t
        # print(np.linalg.norm(Ut.T @ Ut - np.eye(p)))
        Data[1,:,:] = Ut[:,0:p]
    #print(np.linalg.norm(U1 - Ut[:,0:p]))
    tend = time.time()
    Time_data[0,0] = t1-t0
    Time_data[0,1] = t2-t1
    Time_data[0,2] = tend-t2
    print("------------- Times -------------")
    print("Preprocess:  ",t1-t0)
    print("Precompute:  ",t2-t1)
    print("Interpolate: ",tend-t2)
    print("---------------------------------")


    print("*** Running Polar light (no Cayley) retraction ***")

    # Step 1: Obtain Xi
    t0 = time.time()
    Xi = STR.Stiefel_PL_inv_ret(U0,U1)

    # Check that we recover U1
    # print(np.linalg.norm(U1 - STR.Stiefel_PL_ret(U0,Xi)))

    # For the computation
    t1 = time.time()

    A = U0.T @ Xi
    T,W = scipy.linalg.schur(A,output = 'real')
    U0W = U0 @ W

    R = np.linalg.qr(Xi-U0 @ A, mode='r')   
    M, Sing, VT = scipy.linalg.svd(R,\
                                full_matrices=True,\
                                compute_uv=True,\
                                overwrite_a=True)

    XiW = Xi @ W
    WVT = W.T@VT.T
    t2 = time.time()
    for t in I:
        Sing_t = np.sqrt(1/(t**2 * Sing**2 + 1))
        STS_t = (WVT*Sing_t) @ VT
                                            # Pull W.T outside and muliply with VT.T
        Ut = (U0W @ ( (linalg.expm(t*T)-t*T)) + t * XiW) @ STS_t
        #print(np.linalg.norm(Ut.T @ Ut - np.eye(p)))
        Data[2,:,:] = Ut[:,0:p]
    #print(np.linalg.norm(U1 - Ut[:,0:p]))

    tend = time.time()
    Time_data[1,0] = t1-t0
    Time_data[1,1] = t2-t1
    Time_data[1,2] = tend-t2
    print("------------- Times -------------")
    print("Preprocess:  ",t1-t0)
    print("Precompute:  ",t2-t1)
    print("Interpolate: ",tend-t2)
    print("---------------------------------")


    print("*** Running Polar light (with Cayley) retraction ***")

    # Step 1: Obtain Xi
    t0 = time.time()
    Xi = STR.Stiefel_PL_inv_ret(U0,U1,2)

    # Check that we recover U1
    #print(np.linalg.norm(U1 - STR.Stiefel_PL_ret(U0,Xi,2)))

    # For the computation
    t1 = time.time()

    A = U0.T @ Xi
    T,W = scipy.linalg.schur(A,output = 'real')
    U0W = U0 @ W

    R = np.linalg.qr(Xi-U0 @ A, mode='r')   
    M, Sing, VT = scipy.linalg.svd(R,\
                                full_matrices=True,\
                                compute_uv=True,\
                                overwrite_a=True)

    XiW = Xi @ W
    WVT = W.T@VT.T
    t2 = time.time()
    for t in I:
        Sing_t = np.sqrt(1/(t**2 * Sing**2 + 1))
        STS_t = (WVT*Sing_t) @ VT
                                            # Pull W.T outside and muliply with VT.T
        Ut = (U0W @ ( (StAux.Cayley(t*T)-t*T)) + t * XiW) @ STS_t
        #print(np.linalg.norm(Ut.T @ Ut - np.eye(p)))
        Data[3,:,:] = Ut[:,0:p]
    #print(np.linalg.norm(U1 - Ut[:,0:p]))

    tend = time.time()
    Time_data[2,0] = t1-t0
    Time_data[2,1] = t2-t1
    Time_data[2,2] = tend-t2
    print("------------- Times -------------")
    print("Preprocess:  ",t1-t0)
    print("Precompute:  ",t2-t1)
    print("Interpolate: ",tend-t2)
    print("---------------------------------")


    # print("*** Running quasi geodesics (no Cayley) retraction ***")

    # # Step 1: Obtain Xi
    # t0 = time.time()
    # A,B,mBT,C,ABTBC,Q = STR.Stiefel_inv_Quasi_geod(U0,U1,3)

    # #print(np.linalg.norm(ABTBC.T+ABTBC))
    # # Check that we recover U1
    # #print(ABTBC +ABTBC.T)
    # #Urec = np.concatenate((U0,Q),axis = 1) @ linalg.expm(1*ABTBC)
    # #print(np.linalg.norm( Urec[:,0:p].T @ Urec[:,0:p]- np.eye(p)))
    # #print(np.linalg.norm(U1 - Urec[:,0:p]))

    # t1 = time.time()
    # #T,W = scipy.linalg.schur(ABTBC,output = 'real')
    # T,W = scipy.linalg.eig(ABTBC)
    # UQW = np.concatenate((U0,Q),axis = 1) @ W

    # t2 = time.time()

    # for t in I:
    #     #Ut = UQW @ (linalg.expm(t*T) @ np.conj(W.T))
    #     Ut = UQW @ (np.diag(np.exp(t*T)) @ np.conj(W.T))
    #     # print(np.linalg.norm(Ut.T @ Ut - np.eye(p)))
    #     Data[4,:,:] = Ut[:,0:p]
    # print(np.linalg.norm(U1 - Ut[:,0:p]))

    # tend = time.time()
    # Time_data[3,0] = t1-t0
    # Time_data[3,1] = t2-t1
    # Time_data[3,2] = tend-t2
    # print("------------- Times -------------")
    # print("Preprocess:  ",t1-t0)
    # print("Precompute:  ",t2-t1)
    # print("Interpolate: ",tend-t2)
    # print("---------------------------------")


    # print("*** Running quasi geodesics (with Cayley) retraction ***")

    # # Step 1: Obtain Xi
    # t0 = time.time()

    # A,B,mBT,C,ABTBC,Q = STR.Stiefel_inv_Quasi_geod(U0,U1,2)

    # #print(np.linalg.norm(ABTBC.T+ABTBC))
    # # Check that we recover U1
    # #print(ABTBC +ABTBC.T)
    # #Urec = np.concatenate((U0,Q),axis = 1) @ StAux.Cayley(1*ABTBC)
    # #print(np.linalg.norm( Urec[:,0:p].T @ Urec[:,0:p]- np.eye(p)))
    # #print(np.linalg.norm(U1 - Urec[:,0:p]))

    # t1 = time.time()
    # T,W = scipy.linalg.schur(ABTBC,output = 'real')
    # #T,W = scipy.linalg.eig(ABTBC)
    
    # UQW = np.concatenate((U0,Q),axis = 1) @ W
    
    # t2 = time.time()

    # for t in I:
    #     Ut = UQW @ (StAux.Cayley(t*T) @ W.T)
    #     #Ut = UQW @ (StAux.Cayley(t*np.diag(T)) @ np.conj(W.T))
    #     #print(np.linalg.norm(Ut[:,0:p].T @ Ut[:,0:p] - np.eye(p)))
    #     Data[5,:,:] = Ut[:,0:p]
    # tend = time.time()
    # #print(np.linalg.norm(U1 - Ut[:,0:p]))

    # Time_data[4,0] = t1-t0
    # Time_data[4,1] = t2-t1
    # Time_data[4,2] = tend-t2
    # print("------------- Times -------------")
    # print("Preprocess:  ",t1-t0)
    # print("Precompute:  ",t2-t1)
    # print("Interpolate: ",tend-t2)
    # print("---------------------------------")



    # print("*** Running quasi geodesics (with Cayley) retraction ***")

    # # Step 1: Obtain Xi
    # t0 = time.time()

    # A,B,mBT,C,ABTBC,Q = STR.Stiefel_inv_Quasi_geod(U0,U1,2)

    # #print(np.linalg.norm(ABTBC.T+ABTBC))
    # # Check that we recover U1
    # #print(ABTBC +ABTBC.T)
    # #Urec = np.concatenate((U0,Q),axis = 1) @ StAux.Cayley(1*ABTBC)
    # #print(np.linalg.norm( Urec[:,0:p].T @ Urec[:,0:p]- np.eye(p)))
    # #print(np.linalg.norm(U1 - Urec[:,0:p]))

    # t1 = time.time()
    # #T,W = scipy.linalg.schur(ABTBC,output = 'real')
    # T,W = scipy.linalg.eig(ABTBC)
    
    # UQW = np.concatenate((U0,Q),axis = 1) @ W
    
    # t2 = time.time()

    # for t in I:
    #     #Ut = UQW @ (StAux.Cayley(t*T) @ W.T)
    #     Ut = UQW @ (StAux.Cayley(t*np.diag(T)) @ np.conj(W.T))
    #     #print(np.linalg.norm(Ut[:,0:p].T @ Ut[:,0:p] - np.eye(p)))
    #     Data[5,:,:] = Ut[:,0:p]
    # tend = time.time()
    # #print(np.linalg.norm(U1 - Ut[:,0:p]))

    # Time_data[4,0] = t1-t0
    # Time_data[4,1] = t2-t1
    # Time_data[4,2] = tend-t2
    # print("------------- Times -------------")
    # print("Preprocess:  ",t1-t0)
    # print("Precompute:  ",t2-t1)
    # print("Interpolate: ",tend-t2)
    # print("---------------------------------")

    return Time_data
n = 10000
ps = np.array([500,1000,1500,2000])
#ps = np.array([100,200,500,750,1000])
#ps = np.array([1000])
#ps = np.array([10,20,30,40,50])
runtimes = np.zeros([len(ps),5])
processtimes = np.zeros([len(ps),5])
k = 0

for p in ps:
    Ttable = run_timings(n,p)
    print(Ttable[:,2].shape)
    runtimes[k,:] = Ttable[:,2].T
    processtimes[k,:] = Ttable[:,0].T + Ttable[:,1].T 
    k = k + 1

plt.rcParams.update({'font.size': 30})
plt.plot(ps, runtimes[:,0], 'b', marker = "o", markersize = 7, linewidth=2, label = 'PF')
plt.plot(ps, runtimes[:,1], 'g', marker = "v", markersize = 7, linewidth=3, label = 'PL')
plt.plot(ps, runtimes[:,2], 'r', marker = "s", markersize = 7, linewidth=3, label = 'PL Cayley')
# plt.plot(ps, runtimes[:,3], 'c', marker = "x", markersize = 7, linewidth=3, label = 'Qu.Geo.')
# plt.plot(ps, runtimes[:,4], 'm', marker = "d", markersize = 7, linewidth=3, label = 'Qu.Geo. Cayley')

plt.title("Runtime, interpolation")
plt.legend()
plt.xlabel('p')
plt.ylabel('Runtime (s.)')
plt.show()

plt.rcParams.update({'font.size': 30})
plt.plot(ps, processtimes[:,0], 'b', marker = "o", markersize = 7, linewidth=2, label = 'PF')
plt.plot(ps, processtimes[:,1], 'g', marker = "v", markersize = 7, linewidth=3, label = 'PL')
plt.plot(ps, processtimes[:,2], 'r', marker = "s", markersize = 7, linewidth=3, label = 'PL Cayley')
# plt.plot(ps, processtimes[:,3], 'c', marker = "x", markersize = 7, linewidth=3, label = 'Qu.Geo.')
# plt.plot(ps, processtimes[:,4], 'm', marker = "d", markersize = 7, linewidth=3, label = 'Qu.Geo. Cayley')

plt.title("Runtime, preprocessing")
plt.legend()
plt.xlabel('p')
plt.ylabel('Runtime (s.)')
plt.show()