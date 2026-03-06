# Simulate the parametric Fisher KKP equation 
#
# We consider Dirichlet boundary conditions on [-L,L] 
import numpy as np

def fisher_KKP(L,T,Nx,Nt,r,D = 1):

    print("Simulating Fisher KKP with r = " + str(r))
    x = np.linspace(-L,L,num = Nx)
    dx = x[2] - x[1]
    dt = T / Nt

    u = np.zeros([Nx,Nt])

    # At u[:,0] we have the initial condition given by an exponential function over the space
    u[:,0] = np.exp(-((x - 0)**2)/2 )

    # Set up the FD matrix
    Dxx = np.zeros([Nx,Nx])

    for j in range(1,Nx-1):
        Dxx[j,j-1] = 1 / dx ** 2
        Dxx[j,j] = -2 / dx ** 2
        Dxx[j,j+1] = 1 / dx ** 2

    # Boundary conditions
    Dxx[0,0] = -2
    Dxx[0, 1] = 1
    Dxx[Nx-1,Nx-2] = 1
    Dxx[Nx-1,Nx-1] = -2

    for i in range(1,Nt):
        nonlin = r * u[:,i-1] * (1 - u[:,i-1])
        u[:,i] = u[:,i-1] + dt * (D*Dxx @ u[:,i-1] + nonlin)

    return u


