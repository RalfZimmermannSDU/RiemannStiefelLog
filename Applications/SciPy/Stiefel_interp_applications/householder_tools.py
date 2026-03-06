# This script contains routines for preprocessing Stiefel data to 
# in order to create well-conditioned upper p x p blocks

import numpy as np

def housegen(x):
    # Generates a householder vector u, which can produce a Householder matrix Ip - u*u.T
    # Tailored to be used for orthonormal matrices

    u = x.copy()
    mu = np.linalg.norm(x,2)
    if abs(mu) < 1.0e-12:
        print("Possible error in Householder routine")

    u = x / mu
    if (u[0] > 0): 
        u[0] = u[0] + 1
        mu = -mu
    else:
        u[0] = u[0] - 1
    #print(u[1])
    u = u/(np.sqrt(np.abs(u[0])))
    return u, mu

def house_qr(X):
    X = np.copy(X)
    n,k = X.shape

    U = np.zeros([n,k])
    R = np.zeros([k,k])
    Q = np.eye(n)

    for j in range(k):
        U[j:n,j], R[j,j] = housegen(X[j:n,j])
        
        #uu = 
        vT = X[j:n,j+1:k].T @ U[j:n,j]
        vT = vT.T
       
        #vT = np.array(vT)[np.newaxis]
        #print(U)
        out = np.outer(U[j:n,j],vT)
        #X[j:n,j+1:k] = X[j:n,j+1:k] - out
        X[j:n,j+1:k] -= out
        #print( U)
        #print(X[j:n,j+1:k])
        #print(X)
        R[j,j+1:k] = X[j,j+1:k].copy()
        Q = Q @ (np.eye(n) - np.outer(U[:,j],U[:,j]) )
        #print(Q)
    return U, R, Q

def house_block(U):
    n,k = U.shape

    W = np.zeros([n,k])
    Y = np.zeros([n,k])

    W[:,0] = U[:,0]
    Y[:,0] = U[:,0]
    for j in range(1,k):
        z = U[:,j] - W[:,0:j] @ (Y[:,0:j].T @ U[:,j] )
        W[:,j] = z
        Y[:,j] = U[:,j]

    return W,Y

def apply_WYT(X,W,Y):
    # Q^T = (WY^T)^T=YW^T
    QX = X - Y @ (W.T @ X)
    return QX
