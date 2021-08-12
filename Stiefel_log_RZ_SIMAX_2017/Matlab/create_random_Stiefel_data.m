%
%
% @author: Ralf Zimmermann, IMADA, SDU Odense
%
function [U0, U1, Delta] =...
    create_random_Stiefel_data(s, n, p, dist)
%-------------------------------------------------------------
% create a random data set 
% U0, U1 on St(n,p),
%  Delta on T_U St(n,p) with canonical norm 'dist',
% which is also the Riemannian distance dist(U0,U1)
%
% input arguments
%     s = random stream (for reproducability)
% (n,p) = dimension of the Stiefel matrices
% dist  = Riemannian distance between the points U0,U1
%         that are to be created
%
% output arguments
%  U0   = base point on Stiefel mnf
%  U1   = point on Stiefel mnf that is a Riemannian distance of "dist" 
%         away from U0
% Delta = tangent vector such that U1 = Exp_U0(Delta).
%
%-------------------------------------------------------------

%create pseudo-random stiefel matrix U0 via QR of random matrix
X = rand(s, n,p);
[U0,~] = qr(X, 0);

% create random tangent vector in T_U0 St(n,p)
A = rand(s, p,p);
A = A-A';            % random p-by-p skew symmetric matrix
T = rand(s, n,p);

% create tangent vector
Delta = U0*A + T-U0*(U0'*T);

%normalize Delta w.r.t. the canonical metric
norm_Delta = sqrt(trace(Delta'*Delta) - 0.5*trace(A'*A));
Delta = (dist/norm_Delta)*Delta;

% 'project' Delta onto St(n,p) via the Stiefel exponential
U1 = Stiefel_Exp(U0, Delta);

return;
end

