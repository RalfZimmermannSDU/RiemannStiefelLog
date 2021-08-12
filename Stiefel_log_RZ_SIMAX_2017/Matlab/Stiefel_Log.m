% Supplemetary material for the SIMAX manuscript
%
% "A matrix-algebraic algorithm for the Riemannian logarithm on the 
%    Stiefel manifold under the canonical metric", 2017
%
%@author: Ralf Zimmermann, IMADA, SDU Odense
%
function [Delta, conv_hist] = Stiefel_Log(U0, U1, tau, do_Procrustes)
%--------------------------------------------------------------------------
%
% Input arguments      
%  U0, U1 : points on St(n,p)
%     tau : convergence threshold
%
% additional user options:
% do_Procrustes : 0/1 Procrustes preprocessing?
%
% Output arguments
%    Delta : Log^{St}_U0(U1), 
%            i.e. tangent vector such that Exp^St_U0(Delta) = U1
%conv_hist : convergence history
%-------------------------------------------------------------

% check_det     : 0/1 check if initial V0 is in SO(2p)
check_det = 0;

% get dimensions
[n,p] = size(U0);

% store convergence history
conv_hist = [0];

% step 1
M = U0'*U1;

% step 2
[Q,N] = qr(U1 - U0*M,0);   % thin qr of normal component of U1

% step 3
[V, ~] = qr([M;N]);                    % orthogonal completion

% "Procrustes preprocessing"
if do_Procrustes
    [D,S,R]      = svd(V(p+1:2*p,p+1:2*p));
    V(:,p+1:2*p) = V(:,p+1:2*p)*(R*D');
end
V            = [[M;N], V(:,p+1:2*p)];  %          |M  X0|
                                       % now, V = |N  Y0| 
                                  
% check if "V \in SO(2p)"
if check_det
    % ensure that "V \in SO(n)"                                       
    DET = det(V);
    if DET < 0
        V(:,p+1) = (-1)*V(:,p+1);
    end
end
  
% step 4: FOR-Loop
for k = 1:400
    % step 5
    % standard matrix logarithm
    %                 |Ak  -Bk'|
    % compute, LV =   |Bk   Ck |
    [LV, exitflag] = logm(V);
                                  
    % steps 6 - 8: convergence check for lower (pxp)-diagonal block
    C = LV(p+1:2*p, p+1:2*p);
    normC = norm(C, 'fro');
    conv_hist(k) = normC;
    if normC<tau;
        disp(['Stiefel log converged after ', num2str(k),' iterations.']);
        break;
    end
    % step 9: update C
    % standard matrix exponential
    Phi = expm(-C);
    % step 10
    V(:,p+1:2*p) = V(:,p+1:2*p)*Phi;   % update last p columns
end

% prepare output                         |A  -B'|
% upon convergence, we have  logm(V) =   |B   0 | = LV
%     A = LV(1:p,1:p);     B = LV(p+1:2*p, 1:p)
% Delta = U0*A+Q*B
Delta = U0*LV(1:p,1:p) + Q*LV(p+1:2*p, 1:p);
return;
end