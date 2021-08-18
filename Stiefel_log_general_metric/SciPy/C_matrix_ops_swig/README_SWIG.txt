
Compile C module for fast entry-by-entry matrix operations
needed in solving the symmetric Sylvester equation
AX+XA=C
with A symmetric, C skew

This is featured in the function
    def solvsymsyl(A, C):
which is an auxiliary for "Stiefel_Log_alg"
implemented in the file Stiefel_Aux.py


SWIG must be available!

On linux distribution, install SWIG via
***************************************

shell>sudo apt install swig

--
Web lecture on Python plus SWIG:
http://scipy-lectures.github.io/advanced/interfacing_with_c/interfacing_with_c.html



Compile with SWIG via the command line
**************************************

shell>python3 setup_C_matrix_ops.py build_ext -i


To test the method, run the script
**************
shell>python3 script_test_C_matrix_ops_SWIG.py
