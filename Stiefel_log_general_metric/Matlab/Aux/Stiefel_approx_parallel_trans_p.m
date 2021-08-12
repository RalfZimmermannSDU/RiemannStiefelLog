function [A2, R2] = Stiefel_approx_parallel_trans_p(M2,N2, A1, R1, nu)
%-------------------------------------------------------------
% Given Y1,Y2 in St(n,p) and Delta in T_Y1St(n,p),
% approximate the parallel transport of Delta to 
% Delta2 in T_Y2St(n,p)
%
% SPECIAL CASE, WHERE Y1, Y2, Delta can be represented as
%
%    Y1 = U*M1 + Q*N1
%    Y2 = U*M2 + Q*N2U
% Delta = U*A1 + Q*R1
%
% WITH THE SAME FIXED U, Q!
%
% Input: 
% Y2    = U*M2 + Q*N2   : in St(n,p)
% Delta = U*A + Q*R     : in T_Y1 St(n,p)
% nu    = norm of Delta( to be conserved)
%
% Output:
% Delta2 = U*A2 + Q*R2  : in T_Y2 St(n,p)
%-------------------------------------------------------------
epsi = 1.0e-14;

% project Delta onto T_Y2 St(n,p) using the pxp factors only
sym_part = A2sym(M2'*A1 + N2'*R1);

A2 = A1 - M2*sym_part;
R2 = R1 - N2*sym_part;
% rescale
l = sqrt(A2(:)'*A2(:) + R2(:)'*R2(:));

if l > epsi
    A2 = (nu/l)*A2;
    R2 = (nu/l)*R2;
else
    A2 = zeros(size(A1));
    R2 = zeros(size(A1));
end

return;
end