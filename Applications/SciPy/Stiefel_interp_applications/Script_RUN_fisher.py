import scipy.linalg as sc
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import sys

from matplotlib import cm
from matplotlib.colors import LightSource

sys.path.append('../../../Stiefel_log_general_metric/SciPy/')
sys.path.append('../Stiefel_interp/')
sys.path.append('../General_interp_tools/')

import snapshot_analytic_mat    as snam
import Stiefel_interp_funcs     as sifs
import Stiefel_Exp_Log          as EL
import Hermite_interp as HI
import simulate_fisher as simfish
import householder_tools as HST

r = 0.3
L = 30
T = 10 # end time
Nx = 100
Nt = 10000

rank = 6

print("Compute rank " + str(rank) + " POD bases of the Fisher KKP equation for varying r")

#rs = np.array([0.5,0.9])
rs = np.array([0.1,0.5,0.9])
#rs = np.array([0.1,0.3,0.5,0.7,0.9])

Us = np.zeros([len(rs),Nx,rank])
print("----------------------------------------------")
#rs = np.array([0.1,0.5,0.9])

for i in range(len(rs)):
    y = simfish.fisher_KKP(L,T,Nx,Nt,rs[i])

    U, S, VT = sc.svd(y,\
                        full_matrices=False,\
                        compute_uv=True,\
                        overwrite_a=True)
    
    # Truncate
    Us[i,:,:] = U[:,0:rank]

# Align the coordinates wrt. U0 = Us[0,:,:]
sign = False
if sign:
    ref_for_sign = 1
    U0 = np.copy(Us[ref_for_sign,:,:])
    for i in range(len(rs)):
        
        Coord = Us[i,:,:].T @ U0
        Csign = np.diag(np.sign(np.diag(Coord)))
        Us[i,:,:] = Us[i,:,:] @ Csign
        


# for i in [0,2]#range(0,len(rs)):
#     Coord = Us[i,:,:].T @ U0
#     Csign = np.diag(np.sign(np.diag(Coord)))
#     Us[i,:,:] = Us[i,:,:] @ Csign


# print(EL.distStiefel(Us[0,:,:],Us[-1,:,:]))

# Check how the Stiefel logarithm behaves
# print("----------------------------------------------")
# print("Riemannian distance under the Euclidean metric")
# print("U0 is the basis obtained for r = " + str(rs[0]))
# print("Between U0 to U1: " + str(EL.distStiefel(Us[0,:,:] ,Us[1,:,:] ,-0.5)))
# print("Between U0 to U2: " + str(EL.distStiefel(Us[0,:,:] ,Us[2,:,:] ,-0.5)))
# print("----------------------------------------------")

#A = np.array([[1/np.sqrt(2),0],[0,1],[1/np.sqrt(2),0]],dtype=float)
U,R,Q = HST.house_qr(Us[0,:,:])

#U,R = HST.house_qr(A)
W,Y = HST.house_block(U)

Us_c = np.copy(Us)
for i in range(len(rs)):
    Us_c[i,:,:] = Us[i,:,:]#HST.apply_WYT(Us[i,:,:] ,W,Y)
    


#Us[2,:,:] = HST.apply_WYT(Us[2,:,:] ,W,Y)

# # Orthogonality checks 
# print(np.linalg.norm(Us[0,:,:].T @ Us[0,:,:] - np.eye(rank),'fro'))
# print(np.linalg.norm(Us[1,:,:].T @ Us[1,:,:] - np.eye(rank),'fro'))
# print(np.linalg.norm(Us[2,:,:].T @ Us[2,:,:] - np.eye(rank),'fro'))



#Nd = 21 # number of data samples
#ran = np.linspace(rs[0],rs[-1],Nd)
Nd = 81
ran = np.linspace(rs[0],rs[-1],Nd)
#ran = np.linspace(0.1,0.6,Nd)


di = np.zeros([Nd,1])
sigmap_s = np.zeros([3,Nd])
# Create reference data
create_ref_data = 0
if create_ref_data:
    Uref = np.zeros([Nd,Nx,rank])
    Uref_c = np.zeros([Nd,Nx,rank])
    #Uref[0,:,:] = U0
    #Uref_c[0,:,:] = Us_c[0,:,:] # To simplify the code when we addi tionally have to align the signs and perform centering

    for i in range(0,Nd):
        y = simfish.fisher_KKP(L,T,Nx,Nt,ran[i])

        U, S, VT = sc.svd(y,\
                            full_matrices=False,\
                            compute_uv=True,\
                            overwrite_a=True)
        
        # align signs 
        sigmap_s[:,i] = S[rank-2:rank+1]
        
        if sign:
            Coord = U[:,0:rank].T @ U0
            Csign = np.diag(np.sign(np.diag(Coord)))

            Uref[i,:,:] = U[:,0:rank] @ Csign;
            Uref_c[i,:,:] = Uref[i,:,:]#HST.apply_WYT(Uref[i,:,:] ,W,Y);
            print(np.diag(Csign))

        else:
            Uref[i,:,:] = U[:,0:rank] 
            Uref_c[i,:,:] = Uref[i,:,:]#HST.apply_WYT(Uref[i,:,:] ,W,Y);

        
        # Track Riemannian distance from U0
        # di[i] = EL.distStiefel(U0,Uref[i,:,:])
    np.save("Uref",Uref)
    np.save("Uref_c",Uref_c)
    # Simulate system and obtain the basis
else:
    Uref = np.load("Uref.npy")
    Uref_c = np.load("Uref_c.npy")

plotSigma_p = False
if plotSigma_p:
    plt.rcParams.update({'font.size': 20})
    # for i in range(4):
    #     line_sigma  = plt.plot(ran, sigmap_s[i,:], linewidth=3, label = '$\sigma$' + str(i))
    line_sigma  = plt.plot(ran, sigmap_s[0,:], 'k--',linestyle =':', linewidth=3, label = '$\sigma_{p-1}$  ')
    line_sigma  = plt.plot(ran, sigmap_s[1,:], 'b-', linewidth=3, label = '$\sigma_{p}$')
    line_sigma  = plt.plot(ran, sigmap_s[2,:], 'r--.', linewidth=3, label = '$\sigma_{p+1}$')

    plt.legend()
    plt.xlabel('r')
    plt.ylabel('$\sigma$')
    plt.show()

Err = np.zeros([3,len(ran)])
alpha  = -0.5
retra = 3

# Interpolate using the Polar light
Deltas = sifs.Stiefel_geodesic_interp_pre(Us_c,\
                                            rs,\
                                            alpha,\
                                            retra)

U1 = Us_c[0,0:rank,0:rank]

for i in range(len(rs)-1):
    Delta = Deltas[i,:,:]
    A = Us_c[i,:,:].T @ Delta
    B = Delta - Us_c[i,:,:] @ A
    A_size = np.linalg.norm(A,'fro')
    B_size = np.linalg.norm(B,'fro')

    #print("F-norm of A: " + str(A_size))
    #print("F-norm of B: " + str(B_size))

# for k in range(len(ran)):
#     rs_star = ran[k]
#     U_star = sifs.Stiefel_geodesic_interp(Us_c,\
#                                             Deltas,\
#                                             rs,\
#                                             rs_star,\
#                                             alpha,
#                                             retra)
    
#     Err[retra-1,k] = np.linalg.norm( U_star - Uref_c[k,:,:],'fro') / np.linalg.norm(Uref_c[k,:,:],'fro')

# Interpolate using Riemann normal coords and Polar factor retraction
comp_time = np.zeros([3,2])
for d in range(100):
    for r in range(1,4):
        retra = r
        t_start = time.time()
        Deltas = sifs.Stiefel_geodesic_interp_pre(Us,\
                                                    rs,\
                                                    alpha,\
                                                    retra)
        comp_time[r-1,0] = comp_time[r-1,0] + (time.time()-t_start)

        t_start = time.time()
        for k in range(len(ran)):    
            rs_star = ran[k]
            U_star = sifs.Stiefel_geodesic_interp(Us,\
                                                    Deltas,\
                                                    rs,\
                                                    rs_star,\
                                                    alpha,
                                                    retra)
            
            #Err[r-1,k] = np.linalg.norm( U_star - Uref[k,:,:],'fro') / np.linalg.norm(Uref[k,:,:],'fro')
        comp_time[r-1,1] = comp_time[r-1,1] + (time.time()-t_start)

        # print('  ***   ')
        # print('Interpolation of ',len(ran),' data points took ', t_end-t_start, 's')
        # print('  ***   ')


print(comp_time/100)
tim = time.time()
U_star = sifs.Stiefel_geodesic_interp(Us,\
                                                    Deltas,\
                                                    rs,\
                                                    rs_star,\
                                                    alpha,
                                                    retra)

print("max errors:")

print("PW linear, retra1 (Riemann     ):", Err[0,:].max())
print("PW linear, retra2 (polar factor):", Err[1,:].max())
print("PW linear, retra3 (polar light ):", Err[2,:].max())  

do_plot = True
if do_plot:
    plt.rcParams.update({'font.size': 30})
  
    line_RBF1,  = plt.plot(ran, Err[0,:], 'b',linestyle =':', linewidth=3, label = 'Errors Riemann')
    line_RBF2,  = plt.plot(ran, Err[1,:], 'r-', linewidth=3, label = 'Errors PF')
    line_RBF3,  = plt.plot(ran, Err[2,:], 'k--.', linewidth=3, label = 'Errors PL')
    #line_pts, = plt.plot(mu_samples, np.zeros((len(mu_samples),)),  'bo', linewidth=3, label = 'fk')
    plt.legend(loc = 2)
    plt.xlabel('r')
    plt.ylabel('Errors')
    plt.show()

#print(np.linalg.norm(Us[1,:,:] - Uref_c[1,:,:]))

# Plot
# x = np.linspace(-L,L,num = Nx)
# t = np.linspace(0,T,Nt)

# t, x = np.meshgrid(t, x)

# fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
# ls = LightSource(270, 45)
# rgb = ls.shade(y, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
# surf = ax.plot_surface(x, t, y, rstride=1, cstride=1, facecolors=rgb,
#                        linewidth=0, antialiased=False, shade=False)
# plt.show()