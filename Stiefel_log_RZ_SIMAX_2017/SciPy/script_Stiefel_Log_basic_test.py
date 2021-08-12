#------------------------------------------------------------------------------
# Basic test for Stiefel log:
#
# Compute the average iteration count for the Stiefel Log for
# input data that are a preselected Riemannian distance apart.
#
#
# as featured in
# R. Zimmermann: "A matrix-algebraic algorithm for the Riemannian logarithm
# on the Stiefel manifold under the canonical metric."
# SIAM Journal on Matrix Analysis and Applications, 38(2):322-342, 2017.
#
# @author: Ralf Zimmermann, IMADA, SDU Odense
#------------------------------------------------------------------------------

import scipy
from scipy import linalg

import matplotlib.pyplot as plt

# local module
import Stiefel_Exp_Log as StEL

print('Running script_Stiefel_Log_basic_test.py')

# set dimensions
n = 1000
p = 50
# set number of random experiments
runs = 10
dist = 0.8*scipy.pi

# initialize 
average_iters = 0.0
for j in range(runs):
    #create random stiefel data
    U0, U1, Delta = StEL.create_random_Stiefel_data(n, p, dist)
    # compute the Stiefel logarithm
    Delta_rec, conv_hist = StEL.Stiefel_Log_alg(U0, U1, 1.0e-13)
    # Check the numerical accuracy of ||Log_U(Exp_(Delta)) - Delta||
    print('Recovery up to a numerical accuracy of:',\
          linalg.norm(Delta_rec - Delta, 'fro'))
    average_iters = average_iters + len(conv_hist)
 # end for...
    
average_iters = average_iters/runs
print('The average iteration count of the Stiefel log is ', average_iters)



#plot the convergence history of the last run:
# create plot object 
fig, ax1 = plt.subplots()

# plot convergence history
ax1.semilogy(range(len(conv_hist)), conv_hist, 'ko-')
ax1.set_title("Convergence history (last run)")

plt.xlabel("iterations")
plt.ylabel("||C||")

# execute the plot
plt.show()
