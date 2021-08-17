#------------------------------------------------------------------------------
#
# Basic test for Stiefel log:
#
# Compute the average iteration count for the Stiefel Log for
# input data that are a preselected Riemannian distance apart.
#
#
# This script is an illustrative example for 
# solving the local geodesic endpoint problem
#    = computing the Riemannian logarithm on the Stiefel manifold.
#
# The algorithms work for a one-parameter family of metrics, including the 
# Euclidean and the canonical metric.
# The canonical metric allows for special algorithmic treatment.
# For all other metrics, a tailored shooting method is invoked.
#
# For theoretical background and description of the algorithms, see
#
# R. Zimmermann, K. H\"uper.
# "Computing the Riemannian logarithm on the Stiefel manifold: 
#  metrics, methods and performance", arXiv:2103.12046, March 2022
#
# If you make use of these methods, please cite the aforementioned reference.
#
#
# @author: Ralf Zimmermann, IMADA, SDU Odense
#------------------------------------------------------------------------------

import scipy
from scipy import linalg
import matplotlib.pyplot as plt
# local module
import Stiefel_Exp_Log as StEL

print('Running script_Stiefel_Log_basic_test.py')

# *** BEGIN: USER PARAMETERS ***
# set dimensions
n = 100
p = 40
# choose metric parameter: alpha = -0.5: Euclidean, alpha = 0.0: canonical
alpha = -0.0
# numerical convergence threshold
tau = 1.0e-12
# set number of random experiments
runs = 10
# set distance of Stiefel points
dist = 0.7*scipy.pi
# plot convergence history of last run?
do_plot = True
# *** END: USER PARAMETERS ***


# initialize iteration counter
average_iters = 0.0
# initialize accuracy indicator
average_acc   = 0.0
for j in range(runs):
    #create random Stiefel data
    U0, U1, Delta = StEL.create_random_Stiefel_data(n, p, dist, alpha)
    
    # compute the Stiefel logarithm
    Delta_rec, conv_hist = StEL.Stiefel_Log(U0, U1, tau, alpha); 
    # check, if Stiefel logarithm recovers Delta
    num_acc = linalg.norm(Delta_rec - Delta, 'fro')/linalg.norm(Delta, 'fro')
    print('recovery up to:', num_acc)
    average_iters = average_iters + len(conv_hist)
    average_acc   = average_acc + num_acc
#end for...
    
average_iters = average_iters/runs
average_acc   = average_acc/runs
print('The average iteration count of the Stiefel log is ', average_iters)
print('The average relative accuracy of the Stiefel log is ', average_acc)

if do_plot:
    #plot the convergence history of the last run:
    # create plot object 
    fig, ax1 = plt.subplots()
    
    # plot convergence history
    ax1.semilogy(range(len(conv_hist)), conv_hist, 'ko-')
    ax1.set_title("Convergence history (last run)")
    
    plt.xlabel("iterations")
    plt.ylabel("error")
    
    # execute the plot
    plt.show()