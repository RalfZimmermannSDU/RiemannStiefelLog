
Matlab code for the Riemannian logarithm
on the Stiefel manifold under the canonical metric

The algorihm solves the "geodesic endpoint problem" locally.


Execute the script
>>script_Stiefel_Log_basic_test

This script conducts the canonical Stiefel logarithm 
for artifical pseudo-random Stiefel data.
It performs 10 runs and outputs the average iteration count.

The functions are contain in 

The paramter settings are given in the script 
"script_Stiefel_Log_basic_test.m".
Edit lines 12-21 to change
* the dimension
* the number of runs
* the distance of the Stiefel data
* the numerical target accuracy

If you use these algorithms, please cite

R. Zimmermann, "A matrix-algebraic algorithm for the Riemannian logarithm
on the Stiefel manifold under the canonical metric."
SIAM Journal on Matrix Analysis and Applications, 38(2):322-342, 2017.