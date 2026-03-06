import scipy.linalg as sc
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import sys

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

Us = np.zeros([3,Nx,rank])
print("----------------------------------------------")
#rs = np.array([0.1,0.5,0.9])
rs = np.array([0.1,0.5])
for i in range(len(rs)):
    y = simfish.fisher_KKP(L,T,Nx,Nt,rs[i])

    U, S, VT = sc.svd(y,\
                        full_matrices=False,\
                        compute_uv=True,\
                        overwrite_a=True)
    
    # Truncate
    Us[i,:,:] = U[:,0:rank]

# Align the coordinates wrt. U0 = Us[0,:,:]
U0 = np.copy(Us[0,:,:])

for i in range(1,len(rs)):
    Coord = Us[i,:,:].T @ U0
    Csign = np.diag(np.sign(np.diag(Coord)))
    Us[i,:,:] = Us[i,:,:] @ Csign


# Check how the Stiefel logarithm behaves
print("----------------------------------------------")
print("Riemannian distance under the Euclidean metric")
print("U0 is the basis obtained for r = " + str(rs[0]))
print("Between U0 to U1: " + str(EL.distStiefel(Us[0,:,:] ,Us[1,:,:] ,-0.5)))
#print("Between U0 to U2: " + str(EL.distStiefel(Us[0,:,:] ,Us[2,:,:] ,-0.5)))
print("----------------------------------------------")

#A = np.array([[1/np.sqrt(2),0],[0,1],[1/np.sqrt(2),0]],dtype=float)
U,R,Q = HST.house_qr(Us[0,:,:])

#U,R = HST.house_qr(A)
W,Y = HST.house_block(U)

Us_c = np.copy(Us)
Us_c[0,:,:] = HST.apply_WYT(Us[0,:,:] ,W,Y)
Us_c[1,:,:] = HST.apply_WYT(Us[1,:,:] ,W,Y)

#Us[2,:,:] = HST.apply_WYT(Us[2,:,:] ,W,Y)

# # Orthogonality checks 
# print(np.linalg.norm(Us[0,:,:].T @ Us[0,:,:] - np.eye(rank),'fro'))
# print(np.linalg.norm(Us[1,:,:].T @ Us[1,:,:] - np.eye(rank),'fro'))
# print(np.linalg.norm(Us[2,:,:].T @ Us[2,:,:] - np.eye(rank),'fro'))

# Polar factor eigenvalues
U1 = Us_c[1,0:rank,0:rank]
R = U1 @ np.linalg.inv(sc.sqrtm(U1.T @ U1))


Nd = 81 # number of data samples
ran = np.linspace(0.1,0.5,Nd)


# Create reference data
create_ref_data = 0
if create_ref_data:
    Uref = np.zeros([Nd,Nx,rank])
    Uref_c = np.zeros([Nd,Nx,rank])
    Uref[0,:,:] = U0
    Uref_c[0,:,:] = Us_c[0,:,:] # To simplify the code when we additionally have to align the signs and perform centering

    for i in range(1,Nd):
        y = simfish.fisher_KKP(L,T,Nx,Nt,ran[i])

        U, S, VT = sc.svd(y,\
                            full_matrices=False,\
                            compute_uv=True,\
                            overwrite_a=True)
        
        # align signs -> map to centered coordinates
        Uref[i,:,:] = U[:,0:rank] @ Csign;
        Uref_c[i,:,:] = HST.apply_WYT(Uref[i,:,:] ,W,Y);

    np.save("Uref",Uref)
    np.save("Uref_c",Uref_c)
    # Simulate system and obtain the basis
else:
    Uref = np.load("Uref.npy")
    Uref_c = np.load("Uref_c.npy")


# Errors
Err = np.zeros([3,len(ran)])


alpha  = -0.5
retra = 3
Deltas = sifs.Stiefel_geodesic_interp_pre(Us_c,\
                                            rs,\
                                            alpha,\
                                            retra)


for k in range(len(ran)):
    rs_star = ran[k]
    U_star = sifs.Stiefel_geodesic_interp(Us_c,\
                                            Deltas,\
                                            rs,\
                                            rs_star,\
                                            alpha,
                                            retra)
    
    Err[retra-1,k] = np.linalg.norm( U_star - Uref_c[k,:,:],'fro') / np.linalg.norm(Uref_c[k,:,:],'fro')


for r in range(1,3):
    retra = r

    Deltas = sifs.Stiefel_geodesic_interp_pre(Us,\
                                                rs,\
                                                alpha,\
                                                retra)

    
    #for k in range(len(ran)):
    for k in range(len(ran)):    
        rs_star = ran[k]
        U_star = sifs.Stiefel_geodesic_interp(Us,\
                                                Deltas,\
                                                rs,\
                                                rs_star,\
                                                alpha,
                                                retra)
        
        Err[r-1,k] = np.linalg.norm( U_star - Uref[k,:,:],'fro') / np.linalg.norm(Uref[k,:,:],'fro')
#print(Uref[0,:,:].T @ Uref[0,:,:])
print("max errors:")

print("PW linear, retra1 (Riemann     ):", Err[0,:].max())
print("PW linear, retra2 (polar factor):", Err[1,:].max())
print("PW linear, retra3 (polar light ):", Err[2,:].max())  


do_plot = True
if do_plot:
    plt.rcParams.update({'font.size': 30})
  
    line_RBF1,  = plt.plot(ran, Err[0,:], 'r-.', linewidth=3, label = 'PW lin, retra1:Riemann')
    line_RBF2,  = plt.plot(ran, Err[1,:], 'k-.', linewidth=3, label = 'PW lin, retra2:PF')
    line_RBF3,  = plt.plot(ran, Err[2,:], 'b-.', linewidth=3, label = 'PW lin, retra3:PL')
    #line_pts, = plt.plot(mu_samples, np.zeros((len(mu_samples),)),  'bo', linewidth=3, label = 'fk')
    plt.legend()
    plt.xlabel('mu')
    plt.ylabel('Errors')
    plt.show()

#print(np.linalg.norm(Us[1,:,:] - Uref_c[1,:,:]))

# # Plot
# x = np.linspace(-L,L,num = Nx)
# t = np.linspace(0,T,Nt)

# t, x = np.meshgrid(t, x)

# fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
# ls = LightSource(270, 45)
# rgb = ls.shade(y, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
# surf = ax.plot_surface(x, t, y, rstride=1, cstride=1, facecolors=rgb,
#                        linewidth=0, antialiased=False, shade=False)
# plt.show()