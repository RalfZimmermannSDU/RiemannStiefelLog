This folder contains tools for conducting interpolation on the compact Stiefel manifold.

It relies on the SciPy implementation of the Riemannian Stiefel Exp and Log maps from
"RiemannStiefelLog/Stiefel_log_general_metric/SciPy".

In particular, it includes the academic examples of interpolating the orthogonal matrix factors 
of a parametric QR-Decomposition and a parametric Singular Value Decomposition as featured in
the paper:

R. Zimmermann:
HERMITE INTERPOLATION AND DATA PROCESSING ERRORS ON RIEMANNIAN MATRIX MANIFOLDS
SIAM J. SCI . COMPUT. Vol. 42, No. 5, pp. A2593--A2619



The folder contains the following subfolders
"General_interp_tools" : 
  This folder contains tools for 
    * cubic Hermite interpolation
    * Radial Basis Function (RBF) interpolation
    * cubic spline intepolation
  for data on a vector space
  
"Stiefel_interp" : 
  This folder contains the Riemannian counterparts for
    * geodesic interpolation
    * cubic Hermite interpolation
    * Radial Basis Function (RBF) interpolation
    * cubic spline intepolation
  for data on the curved Riemannian Stiefel manifold.
  
"Stiefel_interp_applications" : 
  This folder contains executable examples of interpolation problems with Stiefel manifold data.
  More precisely, it features the python3-scripts
  * script_Hermite_St_Interp_SISC_Section5_2.py
  * script_Hermite_St_Interp_SISC_Section5_3.py
  * script_Hermite_St_Interp_SISC_Section5_4.py
  * script_Hermite_St_Interp_SISC_test_eq25_Section_S5.py
  that correspond to the numerical experiments featured in Sections 5.2, 5.3, 5.4 of the above SISC paper.
  
  For example, in a linux shell, execute
  >>python3 script_Hermite_St_Interp_SISC_Section5_2.py


Make sure that the folder structure aligns with the data paths.

If you are using the code, please consider citing the associated references:

R. Zimmermann
A MATRIX-ALGEBRAIC ALGORITHM FOR THE RIEMANNIAN LOGARITHM ON THE STIEFEL MANIFOLD UNDER THE CANONICAL METRIC
SIAM J. MATRIX ANAL. APPL. Vol. 38, No. 2, pp. 322–342, 2017

R. Zimmermann:
HERMITE INTERPOLATION AND DATA PROCESSING ERRORS ON RIEMANNIAN MATRIX MANIFOLDS
SIAM J. SCI . COMPUT. Vol. 42, No. 5, pp. A2593--A2619, 2020


R. Zimmermann, K. H\"uper:
Computing the Riemannian logarithm on the Stiefel manifold: metrics, methods and performance 
arXiv:2103.12046, March 2022


zimmermann@imada.sdu.dk

