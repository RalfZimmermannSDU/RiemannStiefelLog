%--------------------------------------------------------------------------
% Stiefel exponential for the canonical metric
%
% A. Edelman, T. A. Arias, and S. T. Smith. 
% "The geometry of algorithms with orthogonality constraints." 
% SIAM Journal on Matrix Analysis and Applications, 20(2):303-353, 1998.
%
%
%
%@author: Ralf Zimmermann, IMADA, SDU Odense
function [U1] = Stiefel_Exp(U0, Delta)
%--------------------------------------------------------------------------
% Input arguments      
% U0    : base point on St(n,p)
% Delta : tangent vector in T_U0 St(n,p)

% Output arguments
%   U1    : Exp^{St}_U0(Delta), 
%--------------------------------------------------------------------------
% get dimensions
[n,p] = size(U0);
A = U0'*Delta;                                       % horizontal component
A = 0.5*(A-A');                                          % ensure A is skew

K = Delta-U0*A;                                          % normal component
[Qe,Re] = qr(K, 0);                                % qr of normal component


% matrix exponential
[[A, -Re'];[Re, zeros(p)]];
exp_part = expm([[A, -Re'];[Re, zeros(p)]]);
exp_part = exp_part(:,1:p);


U1 = [U0, Qe]*exp_part;
return;
end
