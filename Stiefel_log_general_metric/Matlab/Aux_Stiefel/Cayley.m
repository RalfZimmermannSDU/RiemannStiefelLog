function [Cay] = Cayley(X)
% classical Cayley trafo
    p = size(X,1);
    Cay = inv(eye(p)-0.5*X)*(eye(p) + 0.5*X);
    return;
end