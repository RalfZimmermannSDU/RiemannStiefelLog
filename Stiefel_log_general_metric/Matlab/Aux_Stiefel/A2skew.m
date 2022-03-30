function [Askew] = A2skew(A)
% extract the skew-symmetric part of A

Askew = 0.5*(A-A');
return;
end

