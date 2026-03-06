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
Nt = 100000

rank = 6

print("Compute rank " + str(rank) + " POD bases of the Fisher KKP equation for varying r")

Us = np.zeros([3,Nx,rank])
print("----------------------------------------------")
rs = np.array([0.1,0.5,0.9])
for i in range(3):
    y = simfish.fisher_KKP(L,T,Nx,Nt,rs[i])

    U, S, VT = sc.svd(y,\
                        full_matrices=False,\
                        compute_uv=True,\
                        overwrite_a=True)
    
    # Truncate
    Us[i,:,:] = U[:,0:rank]

# Align the coordinates wrt. U0 = Us[0,:,:]
U0 = Us[0,:,:]

for i in range(1,3):
    Coord = Us[i,:,:].T @ U0
    Csign = np.diag(np.sign(np.diag(Coord)))
    Us[i,:,:] = Us[i,:,:] @ Csign


# Check how the Stiefel logarithm behaves
print("----------------------------------------------")
print("Riemannian distance under the Euclidean metric")
print("U0 is the basis obtained for r = " + str(rs[0]))
print("Between U0 to U1: " + str(EL.distStiefel(Us[0,:,:] ,Us[1,:,:] ,-0.5)))
print("Between U0 to U2: " + str(EL.distStiefel(Us[0,:,:] ,Us[2,:,:] ,-0.5)))
print("----------------------------------------------")

#A = np.array([[1/np.sqrt(2),0],[0,1],[1/np.sqrt(2),0]],dtype=float)
U,R,Q = HST.house_qr(Us[0,:,:])


#U,R = HST.house_qr(A)
W,Y = HST.house_block(U)

Us[0,:,:] = HST.apply_WYT(Us[0,:,:] ,W,Y)
Us[1,:,:] = HST.apply_WYT(Us[1,:,:] ,W,Y)
Us[2,:,:] = HST.apply_WYT(Us[2,:,:] ,W,Y)

# # Orthogonality checks 
# print(np.linalg.norm(Us[0,:,:].T @ Us[0,:,:] - np.eye(rank),'fro'))
# print(np.linalg.norm(Us[1,:,:].T @ Us[1,:,:] - np.eye(rank),'fro'))
# print(np.linalg.norm(Us[2,:,:].T @ Us[2,:,:] - np.eye(rank),'fro'))

# Polar factor eigenvalues
U1 = Us[2,0:rank,0:rank]
R = U1 @ np.linalg.inv(sc.sqrtm(U1.T @ U1))



#print(np.linalg.norm(Us[i,:,:].T @ Us[i,:,:]  - np.eye(6),'fro'))

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