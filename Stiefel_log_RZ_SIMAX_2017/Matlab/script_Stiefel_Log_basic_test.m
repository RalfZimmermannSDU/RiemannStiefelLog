% Basic test for Stiefel log
%
% Featured in
% "A matrix-algebraic algorithm for the Riemannian logarithm on the 
%    Stiefel manifold under the canonical metric", SIMAX 2017
%
% @author: Ralf Zimmermann, IMADA, SDU Odense
%
clear; close all;


%---User Settings----------------------------------------------------------
% set dimensions
n = 120%200;
p = 6%30;

% set number of random experiments
runs = 1;       % number of runs
dist = 2.0*pi;   % Riemannian distance of test data
tau =  1.0e-11;  % numerical convergence threshold
%---End: User Settings-----------------------------------------------------


% initialize performance indicators
performance_time = 0.0;
number_iters     = 0;
num_accuracy     = 0.0;


for j=1:runs
    %----------------------------------------------------------------------
    %create random stiefel data
    % fix stream of random numbers for reproducability
    s = RandStream('mt19937ar','Seed', 100*j);
    [U0, U1, Delta] = create_random_Stiefel_data(s, n, p, dist);
    %----------------------------------------------------------------------

    %----------------------------------------------------------------------
    % do procrustes preprocessing Y/N?
    do_proc   = 1;
    tic;
    [Delta_rec, conv_hist_alg_log] = Stiefel_Log(U0,...
                                                 U1,...
                                                 tau,...                                              
                                                 do_proc);
    performance_time = performance_time + toc;
    number_iters     = number_iters + length(conv_hist_alg_log);
    num_accuracy     = num_accuracy + norm(Delta_rec-Delta, 'fro');
    %----------------------------------------------------------------------  
     
end

%--------------------------------------------------------------------------
% average time and iteration count
disp(['The average performance time is: ',...
        num2str(performance_time/runs)]);

disp(['The average iteration  count is: ',...
        num2str(number_iters/runs)]);

disp(['The average reconstruction accuracy ||Delta - Log_U0(U1)|| is: ',...
        num2str(num_accuracy/runs)]);



% plot results: of course, only the last run will be shown
figure;
handle = semilogy(1:length(conv_hist_alg_log), conv_hist_alg_log, 'k-*');  
set(handle(1),'linewidth',1, 'MarkerSize', 5);
legend(['Stiefel log'])
