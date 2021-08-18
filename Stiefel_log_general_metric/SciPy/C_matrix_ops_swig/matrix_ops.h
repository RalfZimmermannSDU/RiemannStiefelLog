    
 /* Construct the solution matrix for the symmetric sylvester solver
   "solvsymsyl(A, C)" implemented in "Stiefel_Aux.py"

 Input:  
 C_matrix  : nxn matrix, given as flattened long n*n vector
 L _vector : n-array (in actual application: array of eigenvalues)
 X_matrix : output, nxn matrix, given as flattened long n*n vector
 
   This operation is performed
    for j in range(n):
        for k in range(j+1,n):
            X[j,k] = C2[j,k]/(L[j]+L[k])
            X[k,j] = -X[j,k]
*/      
int symsylv_buildsolmat(double* C_matrix,
                        double* L_vector,
                        double* X_matrix,
                        int n_dim);            
