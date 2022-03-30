Matlab code for the Riemannian logarithm
on the Stiefel manifold for a one-parameter family of metrics

- solves the "geodesic endpoint problem" locally

Execute

>>script_Stiefel_Log_performace_comp

This script conduct the Riemannian Stiefel logarithm 
for artifical pseudo-random Stiefel data and compares the algebraic Stiefel
 log with the p-shooting method for the canonical metric.


The algorithms work for a one-parameter family of metrics, including the 
Euclidean and the canonical metric as special cases.
The canonical metric allows for special algorithmic treatment.
For all other metrics, a tailored shooting method is invoked.

For theoretical background and description of the algorithms, see

R. Zimmermann, K. H\"uper.
"Computing the Riemannian logarithm on the Stiefel manifold: 
 metrics, methods and performance", arXiv:2103.12046, March 2021
The paper has been accepted for publication in SIMAX.

If you make use of these methods, please cite the aforementioned reference.


The main functions are
* create_random_Stiefel_data.m
--> creates a data triple U0, U1, Delta such that U1 = Exp_U0(Delta)

* script_Stiefel_Log_PerformanceComp.m  
--> the main script that performs the algorithmic competition

* Stiefel_Exp.m
--> The Stiefel exponential (for all alpha-metrics)

* Stiefel_Log.m
--> The Stiefel log for the canonical metric

* Stiefel_Log_p_Shooting_uni.m
--> The Stiefel log shooting method (for all alpha metrics)


Auxiliary functions in the folder Aux_Stiefel are
* A2skew.m  
--> returns 0.5*(A-A')

* A2sym.m
--> returns 0.5(A+A')

* Cayley.m
--> Cayley trafo

* Exp4Geo.m
--> computes the pxp-blocks associated with the Riemannian exponential 

* Exp4Geo_pre.m  
--> computes the pxp-blocks associated with the Riemannian exponential
    based on a precomputed eigenvalue decomposition

* solvsymsyl.m
--> sover for the symmetric Sylvester equation

* Stiefel_approx_parallel_trans_p.m
--> conducts an approximate parallel transport restricted to the pxp blocks