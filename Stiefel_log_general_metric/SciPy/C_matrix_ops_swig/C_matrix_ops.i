/*  C code for fast entry-by-entry matrix operations
 *  with using C double arrays as input and
 *  numpy typemaps for SWIG. */

%module C_matrix_ops
%{
    /* the resulting C file should be built as a python extension */
    #define SWIG_FILE_WITH_INIT
    /*  Includes the header in the wrapper code */
    #include "matrix_ops.h"
%}

/*  include the numpy typemaps */
%include "numpy.i"
/*  need this for correct module initialization */
%init %{
    import_array();
%}



/*-----------------------------------------------------------------------------
 * int symsylv_buildsolmat(double* C_matrix,
			             double*  L_vector,
                           double* X_matrix,
                           int n_dim)
 The matrices are inputted flattend as long vectors
 *-----------------------------------------------------------------------------*/
/*  typemaps for the arrays, the last will be modified in-place */
%apply (double* IN_ARRAY1, int DIM1) {(double * C_matrix, int ntotal_C)}
%apply (double* IN_ARRAY1, int DIM1) {(double*  L_vector, int n_L)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* X_matrix, int ntotal_X)}

/*  Wrapper for symsylv_buildsolmat that massages the types */
%inline %{
    /*  takes as input numpy arrays */
    int symsylv_buildsolmat_func(double* C_matrix, int ntotal_C, double* L_vector, int n_L, double* X_matrix, int ntotal_X, int n_dim)
    {
        int k;
        /*  calls the original funcion*/
        k = symsylv_buildsolmat(C_matrix, L_vector, X_matrix, n_dim);
        return(k);
    }
%}






/*  Parse the header file to generate wrappers */
%include "matrix_ops.h"
