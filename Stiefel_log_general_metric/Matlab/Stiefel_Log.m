%------------------------------------------------------------------------------
% associated with the publication
%
% R. Zimmermann, K. H\"uper.
% "Computing the Riemannian logarithm on the Stiefel manifold: 
%  metrics, methods and performance", arXiv:2103.12046, March 2022
%
%
%
% This file implements the following method:
%
%
% * Stiefel_Log_alg(U0, U1, tau, do_Procrustes=0, do_Cayley=0, do_Sylvester=1):
%       - compute Delta = Log^{St}_U0(U1), up to an accuracy of tau
%         via algebraic Stiefel log for the canonical metric
%
%@author: Ralf Zimmermann, IMADA, SDU Odense
% zimmermann@imada.sdu.sdk
function [Delta, conv_hist] = Stiefel_Log(U0, U1, tau,...
                                 do_Procrustes, do_Cayley, do_Sylvester)
%-------------------------------------------------------------
%
% Input arguments      
%  U0, U1 : points on St(n,p)
%     tau : convergence threshold
%
% additional user options:
% do_Procrustes : 0/1 Procrustes preprocessing?
% do_Cayley     : 0/1 Cayley trafo replacing expm?
% do_Sylvester  : 0/1 Sylvester equation for higher-order BCH terms?
%
% Output arguments
%    Delta : Log^{St}_U0(U1), 
%            i.e. tangent vector such that Exp^St_U0(Delta) = U1
%conv_hist : convergence history
%-------------------------------------------------------------

% check_det     : 0/1 check if initial V0 is in SO(2p), costly
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
    Procy = R*D'
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
    
    %[LV, exitflag] = logm(V); % <-- use this line for matlabs matrix-log
    LV = SchurLog(V);
                                  
    % steps 6 - 8: convergence check for lower (pxp)-diagonal block
    C = LV(p+1:2*p, p+1:2*p);
    normC = norm(C, 'fro');
    conv_hist(k) = normC;
    if normC<tau;
        disp(['Stiefel log converged after ', num2str(k),' iterations.']);
        break;
    end
    % step 9: update C
    if do_Sylvester
        % indices of diagonal p by p matrix
        diag_pp = 1:p+1:p*p;
        % set up symmetric Sylvester problem
        % compute (1.0/12.0)*B*B' - 0.5*eye(p);
        % Caution: the block LV(p+1:2*p) contains -B' !
        %          need to correct for the correct sign
        Msym =(-1.0/12.0)*LV(p+1:2*p, 1:p)*LV(1:p, p+1:2*p);
        Msym(diag_pp) = Msym(diag_pp) - 0.5;
        
        % solve Sylvester equation
        %Csylv = sylvester(Msym, Msym, C);  % <-- use this line for matlabs
                                            %     sylvester solver
                                            
        Csylv = solvsymsyl(Msym, C);        % this solver exploits symmetry
        
        % make Csylv exactly skew
        % for both cases do_Sylvester = 0,1,
        % eventually exp(-C) is formed. =>return -Csylv here.
        C = -0.5*(Csylv-Csylv');
    end
    if do_Cayley
        % Cayley approx
        Phi = Cayley(-C);
    else
        % standard matrix exponential
        Phi = expm(-C);
    end
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


%--------------------------------------------------------------------------
%
% local function: efficient Schur log evaluation
%--------------------------------------------------------------------------
function [logV] = SchurLog(V)
%
% Compute logm of ORTHOGONAL matrix via Schur decom
% It returns only those blocks that are needed in Stiefel log
%

% get dimensions
n = size(V, 1);
p = floor(n/2);

% start with Schur decomposition
[Q, S] = schur(V);
% create block-2x2-diagonal matrix
% S must have block diagonal form
logS = gallery('tridiag', zeros(n-1,1), zeros(n,1), zeros(n-1,1));
k = 1;
while k < n+1
    % is block of dim 1x1 => real eigenvalue? 
    if k==n
        if abs(S(k,k) +1.0)<1.0e-12
            disp(['Error: negativ eigval on real axis'])
        end
        k = k+1;
    elseif abs(S(k+1,k))<1.0e-12
        if abs(S(k,k) +1.0)<1.0e-12
            disp(['Error: negativ eigval on real axis'])
        end
        % entry stays zero, just skip ahead
        k = k+1;
    else
        % there is a 2x2 block S(k:k+1, k:k+1)
        % this block must be real orthogonal of the form
        % |cos(phi) sin(phi)|
        % |-sin(phi) cos(phi)|
        logtmp = logm(S(k:k+1, k:k+1));
        phi = logtmp(1,2);
        logS(k,k+1) = phi;
        logS(k+1,k) = -phi;
        k = k+2;
    end
end

logV = Q*(logS*Q');

return;
end





