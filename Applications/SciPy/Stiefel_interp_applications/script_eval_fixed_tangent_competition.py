

#------------------------------------------------------------------------------
# Compare the cost of evolving a retraction as time increases. 
# In such a setting, precomputing several quantities is possible, so that 
# we only have to compute the true t-dependent part of the retractions
#
# We consider:
# Polar factor retraction 
# Polar light retraction w. and wout. Cayley 
# Quasi geodesic w. and wout. Cayley
# QR retraction 
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
9


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
    T,W = scipy.linalg.schur(S,output = 'real')
    #T,W = scipy.linalg.eig(S)
    UQW = UQ @ W



    Time_data = np.zeros([7,3])
    # # ti = time.time()
    # for t in I:
    #     Ut = UQW @ (StAux.BlockExp(t*T) @ W.T)
    #     #Ut = UQ @ scipy.linalg.expm(t*S)
        
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

    print(np.linalg.norm(U1 - Ut[:,0:p]))
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

        Ut = (U0W @ ( (StAux.BlockExp(t*T)-t*T)) + t * XiW) @ STS_t
        #print(np.linalg.norm(Ut.T @ Ut - np.eye(p)))
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

        Ut = (U0W @ ( (StAux.Cayley_Blocked(t*T)-t*T)) + t * XiW) @ STS_t
        #print(np.linalg.norm(Ut.T @ Ut - np.eye(p)))
    print(np.linalg.norm(U1 - Ut[:,0:p]))

    tend = time.time()
    Time_data[2,0] = t1-t0
    Time_data[2,1] = t2-t1
    Time_data[2,2] = tend-t2
    print("------------- Times -------------")
    print("Preprocess:  ",t1-t0)
    print("Precompute:  ",t2-t1)
    print("Interpolate: ",tend-t2)
    print("---------------------------------")


    print("*** Running (Stiefel--like) quasi geodesics (no Cayley) ***")

    # Step 1: Obtain Xi
    t0 = time.time()
    A,B,mBT,C,ABTBC,Q = STR.Stiefel_inv_Quasi_geod(U0,U1,3)

    #print(np.linalg.norm(ABTBC.T+ABTBC))
    # Check that we recover U1
    #print(ABTBC +ABTBC.T)
    #Urec = np.concatenate((U0,Q),axis = 1) @ linalg.expm(1*ABTBC)
    #print(np.linalg.norm( Urec[:,0:p].T @ Urec[:,0:p]- np.eye(p)))
    #print(np.linalg.norm(U1 - Urec[:,0:p]))

    t1 = time.time()
    T,W = scipy.linalg.schur(ABTBC,output = 'real')
    #T,W = scipy.linalg.eig(ABTBC)
    UQW = np.concatenate((U0,Q),axis = 1) @ W

    t2 = time.time()

    for t in I:
        Ut = UQW @ (StAux.BlockExp(t*T) @ W.T)
        #Ut = UQW @ (np.diag(np.exp(t*T)) @ np.conj(W.T))
        # print(np.linalg.norm(Ut.T @ Ut - np.eye(p)))
    print(np.linalg.norm(U1 - Ut[:,0:p]))

    tend = time.time()
    Time_data[3,0] = t1-t0
    Time_data[3,1] = t2-t1
    Time_data[3,2] = tend-t2
    print("------------- Times -------------")
    print("Preprocess:  ",t1-t0)
    print("Precompute:  ",t2-t1)
    print("Interpolate: ",tend-t2)
    print("---------------------------------")


    print("*** Running (Stiefel--like) quasi geodesics (with Cayley) ***")

    # Step 1: Obtain Xi
    t0 = time.time()

    A,B,mBT,C,ABTBC,Q = STR.Stiefel_inv_Quasi_geod(U0,U1,2)

    #print(np.linalg.norm(ABTBC.T+ABTBC))
    # Check that we recover U1
    #print(ABTBC +ABTBC.T)
    #Urec = np.concatenate((U0,Q),axis = 1) @ StAux.Cayley(1*ABTBC)
    #print(np.linalg.norm( Urec[:,0:p].T @ Urec[:,0:p]- np.eye(p)))
    #print(np.linalg.norm(U1 - Urec[:,0:p]))

    t1 = time.time()
    T,W = scipy.linalg.schur(ABTBC,output = 'real')
    #T,W = scipy.linalg.eig(ABTBC)
    
    UQW = np.concatenate((U0,Q),axis = 1) @ W
    
    t2 = time.time()

    for t in I:
        Ut = UQW @ (StAux.Cayley_Blocked(t*T) @ W.T)
        #Ut = UQW @ (StAux.Cayley(t*np.diag(T)) @ np.conj(W.T))
        #print(np.linalg.norm(Ut[:,0:p].T @ Ut[:,0:p] - np.eye(p)))
    tend = time.time()
    print(np.linalg.norm(U1 - Ut[:,0:p]))

    Time_data[4,0] = t1-t0
    Time_data[4,1] = t2-t1
    Time_data[4,2] = tend-t2
    print("------------- Times -------------")
    print("Preprocess:  ",t1-t0)
    print("Precompute:  ",t2-t1)
    print("Interpolate: ",tend-t2)
    print("---------------------------------")

    print("*** Running (Grassmann--like) quasi geodesics ***")

    # Step 1: Obtain Xi
    t0 = time.time()
    Xi = STR.Stiefel_inv_Quasi_geod(U0,U1,1)
    Utilde =  STR.Stiefel_Quasi_geod(U0,Xi)
    print(np.linalg.norm(U1-Utilde))
    #print(np.linalg.norm(ABTBC.T+ABTBC))
    # Check that we recover U1
    #print(ABTBC +ABTBC.T)
    #Urec = np.concatenate((U0,Q),axis = 1) @ linalg.expm(1*ABTBC)
    #print(np.linalg.norm( Urec[:,0:p].T @ Urec[:,0:p]- np.eye(p)))
    #print(np.linalg.norm(U1 - Urec[:,0:p]))

    t1 = time.time()

    A = U0.T @ Xi
    UperpB = Xi - U0 @ A
    Q, S, VT = linalg.svd(UperpB, full_matrices=False,compute_uv=True, overwrite_a=True)

    #T,YY = linalg.schur(A,output='real')


    t2 = time.time()

    for t in I:
        Ut = (U0 @ (VT.T * np.cos(S*t)) + Q * np.sin(S*t)) @ (VT @ linalg.expm(t*A))
        #Ut = UQW @ (np.diag(np.exp(t*T)) @ np.conj(W.T))
        # print(np.linalg.norm(Ut.T @ Ut - np.eye(p)))
    print(np.linalg.norm(U1 - Ut[:,0:p]))

    tend = time.time()
    Time_data[5,0] = t1-t0
    Time_data[5,1] = t2-t1
    Time_data[5,2] = tend-t2
    print("------------- Times -------------")
    print("Preprocess:  ",t1-t0)
    print("Precompute:  ",t2-t1)
    print("Interpolate: ",tend-t2)
    print("---------------------------------")


    print("*** Running QR retraction ***")

    t0 = time.time()
    Xi = STR.Stiefel_QR_inv_ref(U0,U1)
    Utilde,R =  STR.Stiefel_QR_ret(U0,Xi)
    print(np.linalg.norm(U1-Utilde))
    #print(np.linalg.norm(ABTBC.T+ABTBC))
    # Check that we recover U1
    #print(ABTBC +ABTBC.T)
    #Urec = np.concatenate((U0,Q),axis = 1) @ linalg.expm(1*ABTBC)
    #print(np.linalg.norm( Urec[:,0:p].T @ Urec[:,0:p]- np.eye(p)))
    #print(np.linalg.norm(U1 - Urec[:,0:p]))

    


    t2 = time.time()

    for t in I:
        Ut,R = STR.Stiefel_QR_ret(U0,t*Xi)
        #Ut = UQW @ (np.diag(np.exp(t*T)) @ np.conj(W.T))
        # print(np.linalg.norm(Ut.T @ Ut - np.eye(p)))
    print(np.linalg.norm(U1 - Ut[:,0:p]))

    tend = time.time()
    Time_data[6,0] = t1-t0
    Time_data[6,1] = t2-t1
    Time_data[6,2] = tend-t2
    print("------------- Times -------------")
    print("Preprocess:  ",t2-t0)
    print("Precompute:  ", 0 ) 
    print("Interpolate: ",tend-t2)
    print("---------------------------------")

    return Time_data
n = 10000
ps = np.array([500,1000,1500,2000])
#ps = np.array([100,200,500,750,1000])
#ps = np.array([50])
#ps = np.array([10,20,30,40,50])
runtimes = np.zeros([len(ps),7])
processtimes = np.zeros([len(ps),7])
k = 0

for p in ps:
    Ttable = run_timings(n,p)
    runtimes[k,:] = Ttable[:,2].T
    processtimes[k,:] = Ttable[:,0].T + Ttable[:,1].T 
    k = k + 1

np.save('runtimes',runtimes)
np.save('processtimes',processtimes)
# plt.rcParams.update({'font.size': 30})
# my_dpi = 110
# #plt.figure(figsize=(1400/my_dpi, 1500/my_dpi), dpi=my_dpi)
# plt.figure(figsize=(18,14))
# plt.plot(ps, runtimes[:,0], 'b', marker = "o", markersize = 7, linewidth=2, label = 'PF')
# plt.plot(ps, runtimes[:,1], 'g',linestyle =(0, (1, 5)), marker = "v", markersize = 7, linewidth=3, label = 'PL')
# plt.plot(ps, runtimes[:,2], 'r',linestyle = (5, (10, 3)), marker = "s", markersize = 7, linewidth=3, label = 'PL Cayley')
# plt.plot(ps, runtimes[:,3], 'c', linestyle =(0, (5, 5)),marker = "x", markersize = 7, linewidth=3, label = 'Qu.Geo.')
# plt.plot(ps, runtimes[:,4], 'm', linestyle = (0, (3, 10, 1, 10)),marker = "d", markersize = 7, linewidth=3, label = 'Qu.Geo. Cayley')
# plt.plot(ps, runtimes[:,5], 'tab:olive', linestyle = (0, (3, 10, 1, 10)),marker = "d", markersize = 7, linewidth=3, label = 'Qu.Geo. GR-like')


# plt.title("Runtime, interpolation")
# plt.legend(loc = 'upper left')
# plt.xlabel('p')
# plt.ylabel('Runtime (s.)')

# plt.savefig('Interpol_along_rtime.png', dpi=my_dpi)

# plt.show()

# plt.rcParams.update({'font.size': 30})
# #my_dpi = 196
# # plt.figure(figsize=(1400/my_dpi, 1500/my_dpi), dpi=my_dpi)
# plt.figure(figsize=(18,14))
# plt.plot(ps, processtimes[:,0], 'b', marker = "o", markersize = 7, linewidth=2, label = 'PF')
# plt.plot(ps, processtimes[:,1], 'g',linestyle =(0, (1, 5)),  marker = "v", markersize = 7, linewidth=3, label = 'PL')
# plt.plot(ps, processtimes[:,2], 'r', linestyle = (5, (10, 3)), marker = "s", markersize = 7, linewidth=3, label = 'PL Cayley')
# plt.plot(ps, processtimes[:,3], 'c', linestyle =(0, (5, 5)), marker = "x", markersize = 7, linewidth=3, label = 'Qu.Geo.')
# plt.plot(ps, processtimes[:,4], 'm',linestyle = (0, (3, 10, 1, 10)), marker = "d", markersize = 7, linewidth=3, label = 'Qu.Geo. Cayley')
# plt.plot(ps, processtimes[:,5], 'tab:olive', linestyle = (0, (3, 1, 1, 1, 1, 1)),marker = "d", markersize = 7, linewidth=3, label = 'Qu.Geo. GR-like')


# plt.title("Runtime, preprocessing")
# plt.legend(loc = 'upper left')
# plt.xlabel('p')
# plt.ylabel('Runtime (s.)')
# plt.savefig('Interpol_along_preprocess.png', dpi=my_dpi)
# plt.show()