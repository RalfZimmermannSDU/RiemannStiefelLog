%**************************************************************************
% helper function
%
% evaluate the matrix exponential as inherent to the Stiefel geodesics
%
% unified implementation for Euclidean and canonical metric
%
% This function expects the EVD 
% Evecs*Evals*Evecs' = [[f(A), -R'];[R, 0]] as an input.
% Here, f(A) depends on the chosen alpha-metric
%**************************************************************************
function [M,N] = Exp4Geo_pre(t, A_pre, Evecs, evals, alpha)
    p = size(A_pre,1);
    
    %evaluate the matrix exponential, the result must be real
    V = real(Evecs*(exp(t*i*evals).*Evecs(1:p,1:2*p)'));
    if abs(alpha - 0) < 10*eps                           % canonical metric
        M = V(1:p,1:p);
        N = V(p+1:2*p,1:p); 
    elseif abs(alpha+1) > eps
        % If A_pre = (2-x)*A, where x = (2*alpha + 1)/(alpha + 1);
        % then we need y*A,   where y  = (alpha)/(alpha +1);
        % which is given by (y/(1-y))*A_pre = alpha*A_pre    
        Phi = expm(t*alpha*A_pre);
        M = V(1:p,1:p)*Phi;
        N = V(p+1:2*p,1:p)*Phi;
    else
        disp('Error in  Stiefel_Exp: wrong metric. Choose $\alpha \neq -1$ ')
        M = 0; 
        N = 0;
    end
return;
end