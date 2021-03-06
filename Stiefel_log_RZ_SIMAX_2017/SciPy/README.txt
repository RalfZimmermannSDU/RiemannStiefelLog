SciPy code for the Riemannian logarithm
on the Stiefel manifold under the canonical metric.

The algorihm solves the "geodesic endpoint problem" locally.


In a Linux shell, run:

$python3 script_Stiefel_Log_basic_test.py

This script conducts the canonical Stiefel logarithm 
for artifical pseudo-random Stiefel data.
It performs 10 runs and outputs the average iteration count.

The functions are contained in the module "Stiefel_Exp_Log.py".

The parameter settings are given in the script 
"script_Stiefel_Log_basic_test.py".

Edit lines 24-29 to change
* the dimension
* the number of runs
* the distance of the Stiefel input data

If you use these algorithms, please cite

R. Zimmermann, "A matrix-algebraic algorithm for the Riemannian logarithm
on the Stiefel manifold under the canonical metric."
SIAM Journal on Matrix Analysis and Applications, 38(2):322-342, 2017.
