function [ X ] = solvsymsyl(A, C)
%
%
% solve the symmetric sylvester equation
% AX + XA  = C
%
% with A real, symmetric, C skew-sym
%
% via Bathia, Matrix Analysis, Theorem VII.2.3, p. 205
%

% step 1: reduce to diagonal problem A = Q L Q'
%
% AX + XA = C  <=> L Q'XQ + Q'XQ L = Q'C Q
[Q, L] = eig(A);
l = diag(L);
%
C2 = Q'*C*Q;

[n, ~] = size(C);

X = zeros(n);

for j = 1:n
    for k = j+1:n
        X(j,k) = C2(j,k)/(l(j)+l(k));
        X(k,j) = -X(j,k);
    end
end

X = Q*X*Q';
return;

end