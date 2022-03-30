%**************************************************************************
% helper function
%
% evaluate the matrix exponential as inherent to the Stiefel geodesics
%
% unified implementation for all alpha-metrics 
%
%**************************************************************************
function [M,N] = Exp4Geo(A, R, alpha)
    p = size(A,1);
    if abs(alpha - 0) < 10*eps                           % canonical metric
        V = expm([[A, -R'];[R, zeros(p)]]);
        M = V(1:p,1:p);
        N = V(p+1:2*p,1:p); 
    elseif abs(alpha+1) > eps
        x  = (2*alpha + 1)/(alpha + 1);
        y  = (alpha)/(alpha +1);
        V = expm([[(2-x)*A, -R'];[R, zeros(p)]]);
        Phi = expm(y*A);
        M = V(1:p,1:p)*Phi;
        N = V(p+1:2*p,1:p)*Phi;
    else
        disp('Error in  Stiefel_Exp: wrong metric. Choose $\alpha \neq -1$ ')
        M = 0; 
        N = 0;
    end
return;
end