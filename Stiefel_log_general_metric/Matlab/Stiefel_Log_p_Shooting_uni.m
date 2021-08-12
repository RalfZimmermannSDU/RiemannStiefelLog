%
% Shooting method for computing the Stiefel logarithm
% for all alpha-metrics
%
% associated with the publication
%
% R. Zimmermann, K. H\"uper.
% "Computing the Riemannian logarithm on the Stiefel manifold: 
%  metrics, methods and performance", arXiv:2103.12046, March 2022
%
%@author: Ralf Zimmermann, IMADA, SDU Odense
%
function [Delta, conv_hist] = Stiefel_Log_p_Shooting_uni(U0, U1, I_int, tau, alpha)
%-------------------------------------------------------------
%
% Input arguments      
%  U0, U1 : points on St(n,p)
%   I_int : discrete unit time interval for approx parallel trafo
%     tau : convergence threshold
%   alpha : metric parameter
%             use alpha =-1/2 for 'euclid' 
%             use alpha =   0 for 'canonical'
%
% Output arguments
%   Delta    : Log^{St}_U0(U1), 
%              i.e. tangent vector such that Exp^St_U0(Delta) = U1
%  conv_hist : convergence history
%-------------------------------------------------------------
%disp([' '])
%disp([' *** p-Shooting method *** '])
%disp([' '])

[n,p] = size(U0);

% controlling parameter
max_iter = 100;

if length(I_int)>2
    max_iter = 2000;
end

tsteps = length(I_int);

% step 1: compute the fixed coordinates U, Q
M0     = U0'*U1;
[Q,R0] = qr(U1 - U0*M0,0);   % thin qr of normal component of U1

% initial gap vector W = U1-U0 needs not be formed
% W = U0*M0 + Q0*R0 - U0 = U0(M0-I) + Q*R0

% compute norm of W, this matches norm(U0*(M0-I) + Q0*R0, 'fro')
n_M0I = norm(M0-eye(p),'fro');
n_R0  = norm(R0,'fro');   
n_w = sqrt(n_M0I^2 + n_R0^2);

% compute initial shooting vector:
% project gap U1-U0 onto T_U0 St(n,p)
A = A2skew(M0);
R = R0;
%now: Delta = U0*A + Q0*R; no need to form explicitly

% scale Delta to norm of W
n_d = sqrt(A(:)'*A(:) + n_R0^2);
A = (n_w/n_d)*A;
R = (n_w/n_d)*R0;

% initialize array of "geodesic p-factors" at each t in t_int
GeoM = zeros(tsteps, p,p);
GeoN = zeros(tsteps, p,p);
GeoM(1,:,:) = eye(p);


% make sure that the iterations start
j = 0;
conv_hist = [];


while (n_w > tau) && (j<max_iter)
    j = j+1;
    
    % evaluate geodesic factors at the discrete steps
    % for the current Delta = U A + Q R
    if tsteps > 3
       % in this case, it is more efficient to 
       % first compute the EVD of [[A, -R'];[R, 0]] and then 
       % the matrix exp for all t-steps.
       if abs(alpha - 0) < 10*eps                        % canonical metric
           A_pre = A;
       elseif abs(alpha+1) > eps
           x  = (2*alpha + 1)/(alpha +1);
           A_pre = (2-x)*A;
       else
           A_pre = A;
       end
       [Evec, Evals] = eig([[A_pre, -R'];[R, zeros(p)]]);
       % eigenvalues are on complex axis
       evals = imag(diag(Evals));
       for k = 2:tsteps
            [M,N] = Exp4Geo_pre(I_int(k), A_pre, Evec, evals, alpha);
            GeoM(k,:,:) = M;
            GeoN(k,:,:) = N;
        end 
    else    
        for k = 2:tsteps
            [M,N] = Exp4Geo(I_int(k)*A, I_int(k)*R, alpha);
            GeoM(k,:,:) = M;
            GeoN(k,:,:) = N;
        end
    end
    % compute new gap vector 
    % W = Geo(1) - U1 
    %   = U0*Mend + Q0*Nend - U1 = U0*(Mend-M0) + Q0*(Nend-R0)
    %
    A_up = squeeze(GeoM(tsteps,:,:))-M0;
    R_up = squeeze(GeoN(tsteps,:,:))-R0;
    
    % compute norm of W
    % this matches norm(U0*M + Q0*N - U1, 'fro')    
    n_w = sqrt(norm(A_up,'fro')^2 + norm(R_up,'fro')^2);
    
    conv_hist(j) = n_w;
    
    % paralelle translate W to T_U0St(n,p):
    %   project gap onto T_(Geo(end))
    %   and then along the geodesic onto T_(Geo(0))
    for k = tsteps:-1:1
        [A_up, R_up] = Stiefel_approx_parallel_trans_p(squeeze(GeoM(k,:,:)),...
                                                       squeeze(GeoN(k,:,:)),...
                                                       A_up, R_up, n_w);
    end
    
    % update Delta= Delta - W;, using the p-factors only
    A = A - A_up;
    R = R - R_up;
end

% form Delta
Delta = U0*A+Q*R;

if j < max_iter
    disp(['p-Shooting method converged in ', num2str(j), ' iterations']);
else 
    disp(['p-Shooting method did not converge']);
end
    
return;
end
