# import Python modules

import numpy
from numpy import random

import time


#------------------------------------------------------------------------------


n = 2500

#create matrix data
A =  random.rand(n,n)
C =  random.rand(n,n)

A = 0.5*(A+A.T) # make symmetric
C = 0.5*(C-C.T) # make skew

L = random.rand(n,) # test vector

X1 = numpy.zeros((n*n,)) # long vector

X2 = numpy.zeros((n,n))

try:
    # local C-Modul
    import C_matrix_ops
except ImportError:
    print("No C module found: executing python code")
    C_matrix_ops = None


#------------------------------------------------------------------------------
# 1.1. matrix operation in wrapped C code
#------------------------------------------------------------------------------
if C_matrix_ops != None:
    start_time=time.time()
    C_matrix_ops.symsylv_buildsolmat_func(C.flatten(), L, X1, n)
    X1 = X1.reshape((n,n))
    t1 = time.time()-start_time
    print("=> solv sym operation in C took ", t1, "sec.")


#------------------------------------------------------------------------------
# 1.2. Matrix operation directly in Python
#------------------------------------------------------------------------------

#initialize
start_time=time.time()

for j in range(n):
        for k in range(j+1,n):
            X2[j,k] = C[j,k]/(L[j]+L[k])
            X2[k,j] = -X2[j,k]

t2 = time.time()-start_time
print("=> Python operation took ", t2, "sec.")

if C_matrix_ops != None:
    #compare results
    print("Speed up", t2/t1)

    # Check the result
    print('Check if the results match:')
    print(numpy.allclose(X1, X2))






