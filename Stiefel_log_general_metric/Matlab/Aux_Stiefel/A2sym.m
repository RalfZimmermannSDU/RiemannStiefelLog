function [Asym] = A2sym(A)
% extract the symmetric part of A

Asym = 0.5*(A+A');
return;
end
