SciPy code for the Riemannian logarithm
on the Stiefel manifold under the canonical metric

solves the "geodesic endpoint problem" locally

In a Linux shell, run either of the scripts

$python3 script_Stiefel_Log_basic_test.py

or 

$python3 script_Stiefel_Log_performace_comp.py

This scripts conducts the canonical Stiefel logarithm 
for artifical pseudo-random Stiefel data.
It performs 10 runs and outputs the average iteration count.

The functions are contain in the module "Stiefel_Exp_Log.py".

The algorithms work for a one-parameter family of metrics, including the 
Euclidean and the canonical metric.
The canonical metric allows for special algorithmic treatment.
For all other metrics, a tailored shooting method is invoked.

For theoretical background and description of the algorithms, see

R. Zimmermann, K. H\"uper.
"Computing the Riemannian logarithm on the Stiefel manifold: 
 metrics, methods and performance", arXiv:2103.12046, March 2022

If you make use of these methods, please cite the aforementioned reference.
