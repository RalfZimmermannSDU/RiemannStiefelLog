#include <stdio.h>
#include <math.h>






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
                        int n_dim)
{
    int i, j;
    int index;
    double entry;

    for(i=0;i<n_dim;i++)
    {
    /* diagonal index*/
    index = i * n_dim + i;
    X_matrix[index] = 0.0;
        for(j=i+1; j<n_dim;j++)
        {
            /* current index */
            index = i * n_dim + j;
            entry =  C_matrix[index]/(L_vector[i]+L_vector[j]);
            X_matrix[index] = entry;
            /* symmetric entry*/
            index = j * n_dim + i;
            X_matrix[index]  = -entry;
        }
    }    
    return(0);
}



