SciPy code for the Riemannian logarithm
on the Stiefel manifold for a one-parameter family of metrics

- solves the "geodesic endpoint problem" locally

In a Linux shell, run either of the scripts

$python3 script_Stiefel_Log_basic_test.py

or 

$python3 script_Stiefel_Log_performance_comp.py

These scripts conduct the Riemannian Stiefel logarithm 
for artifical pseudo-random Stiefel data.

The functions are contained in the module "Stiefel_Exp_Log.py".

The algorithms work for a one-parameter family of metrics, including the 
Euclidean and the canonical metric as special cases.
The canonical metric allows for special algorithmic treatment.
For all other metrics, a tailored shooting method is invoked.

The algebraic Stiefel log procedure involves solving a symmetric Sylvester equation,
which, in turn, involves an entry-by-entry matrix manipulation,
which is known to be slow in Python.
To accelerate the code, you can build a C-module which is then automatically 
called in Python.
To this end, go to the subfolder "C_matrix_ops_swig" and follow the instructions 
given in "README_SWIG.txt".

For theoretical background and description of the algorithms, see

R. Zimmermann, K. H\"uper.
"Computing the Riemannian logarithm on the Stiefel manifold: 
 metrics, methods and performance", arXiv:2103.12046, March 2022

If you make use of these methods, please cite the aforementioned reference.
