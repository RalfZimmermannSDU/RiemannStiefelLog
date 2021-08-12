%--------------------------------------------------------------------------
% Stiefel exponential for all alpha metrics
%
%@author: Ralf Zimmermann, IMADA, SDU Odense
function [U1] = Stiefel_Exp(U0, Delta, alpha)
%--------------------------------------------------------------------------
% Input arguments      
% U0    : base point on St(n,p)
% Delta : tangent vector in T_U0 St(n,p)
% alpha : metric parameter
%               (special cases: alpha = 0   => canonical metric
%                               alpha =-1/2 => Euclidean metric
% Output arguments
%   U1    : Exp^{St}_U0(Delta), 
%--------------------------------------------------------------------------
% get dimensions
[n,p] = size(U0);
A = U0'*Delta;                                       % horizontal component
A = 0.5*(A-A');                                          % ensure A is skew

K = Delta-U0*A;                                          % normal component
[Qe,Re] = qr(K, 0);                                % qr of normal component

if abs(alpha - 0) < 10*eps                               % canonical metric
    % matrix exponential
    [[A, -Re'];[Re, zeros(p)]];
    exp_part = expm([[A, -Re'];[Re, zeros(p)]]);
    exp_part = exp_part(:,1:p);
elseif abs(alpha+1) > eps
    x  = (2*alpha + 1)/(alpha +1);
    y  = (alpha)/(alpha +1);
    Ar = [[(2-x)*A, -Re'];[Re, zeros(p)]];
    exp_part = expm(Ar);
    expAE = expm(y*A);
    % keep first block column, multiply with expAE
    exp_part = exp_part(:,1:p)*expAE;
else
    disp('Error in  Stiefel_Exp: wrong metric. Choose $\alpha \neq -1$ ')
end

U1 = [U0, Qe]*exp_part;
return;
end
