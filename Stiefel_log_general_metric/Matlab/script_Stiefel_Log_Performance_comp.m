% Basic test for Stiefel log
%
% Featured in
% "A matrix-algebraic algorithm for the Riemannian logarithm on the 
%    Stiefel manifold under the canonical metric", SIMAX 2017
%
% @author: Ralf Zimmermann, IMADA, SDU Odense
%
clear; close all;

% add path to auxiliary Stiefel functions
addpath('Aux_Stiefel/')

% set dimensions
n = 500;
p = 250;
%this is for the canonical metric:
alpha = 0;

% set number of random experiments
runs = 10;
dist = 0.7*pi;
tau =  1.0e-11;

% *Methods*
% 1: algebraic Stiefel log
% 2: algebraic Stiefel log + sylvester
% 3: algebraic Stiefel log + sylvester + Cayley
% 4: Stiefel_Log_p_Shooting
% 5: Stiefel_Log_p_Shooting 4 t-steps


iters_array = zeros(5,runs);
time_array  = zeros(5,runs);
is_equal    = zeros(5,runs);
for j=1:runs
    %----------------------------------------------------------------------
    %create random stiefel data
    % fix stream of random numbers for reproducability
    s = RandStream('mt19937ar','Seed', 100*j);
    [U0, U1, Delta] = create_random_Stiefel_data(s, n, p, dist, alpha);
    %----------------------------------------------------------------------

    %----------------------------------------------------------------------
    % Method 1:
    % basic algebraic Stiefel logarithm
    % alg settings
    do_proc   = 0;
    do_cayley = 0;
    do_sylv   = 0;
    tic;
    [Delta_rec, conv_hist_alg_log] = Stiefel_Log(U0,...
                                                 U1,...
                                                 tau,...                                              
                                                 do_proc,...
                                                 do_cayley,...
                                                 do_sylv);
    time_array(1,j)  = toc;
    iters_array(1,j) = length(conv_hist_alg_log);
    is_equal(1,j)    = norm(Delta_rec-Delta, 'fro');
    %----------------------------------------------------------------------
    
    
    %----------------------------------------------------------------------
    % Method 2:
    % algebraic Stiefel logarithm + sylvester enhancement
    %    
    % alg settings
    do_proc   = 0;
    do_cayley = 0;
    do_sylv   = 1;
    tic;
    [Delta_rec, conv_hist_alg_log_sylv] = Stiefel_Log(U0,...
                                                      U1,...
                                                      tau,...
                                                      do_proc,...
                                                      do_cayley,...
                                                      do_sylv);
    time_array(2,j)  = toc;
    iters_array(2,j) = length(conv_hist_alg_log_sylv);
    is_equal(2,j)    = norm(Delta_rec-Delta, 'fro');
    %----------------------------------------------------------------------   
    
    
    %----------------------------------------------------------------------
    % Method 3:
    % algebraic Stiefel logarithm + sylvester enhancement + Cayley accel. 
    %     
    % alg settings
    do_proc   = 0;
    do_cayley = 1;
    do_sylv   = 1;
  
    tic;
    [Delta_rec, conv_hist_alg_log_sylvC] = Stiefel_Log(U0,...
                                                       U1,...
                                                       tau,...
                                                       do_proc,...
                                                       do_cayley,...
                                                       do_sylv);
    time_array(3,j)  = toc;
    iters_array(3,j) = length(conv_hist_alg_log_sylvC);
    is_equal(3,j)    = norm(Delta_rec-Delta, 'fro');
    %----------------------------------------------------------------------   
   

    %----------------------------------------------------------------------
    % Method 4:
    % execute shooting method on two steps in [0,1]
    t_steps =  linspace(0.0,1.0,2);
    tic;
    [Delta_rec, conv_hist_pS] = Stiefel_Log_p_Shooting_uni(U0,...
                                                           U1,...
                                                           t_steps,...
                                                           tau,...
                                                           alpha);  
    time_array(4,j)  = toc;
    iters_array(4,j) = length(conv_hist_pS);
    is_equal(4,j)    = norm(Delta_rec-Delta, 'fro'); 
    %----------------------------------------------------------------------
    
    
    %----------------------------------------------------------------------
    % Method 5:
    % execute shooting method on four steps in [0,1]
    t_steps =  linspace(0.0,1.0,4);
    tic;
    [Delta_rec, conv_hist_pSu] = Stiefel_Log_p_Shooting_uni(U0,...
                                                            U1,...
                                                            t_steps,...
                                                            tau,...
                                                            alpha);
    time_array(5,j)  = toc;
    iters_array(5,j) = length(conv_hist_pSu);
    is_equal(5,j)    = norm(Delta_rec-Delta, 'fro');
    %----------------------------------------------------------------------
end

%--------------------------------------------------------------------------
% average time and iteration count
disp(['The average timing of the various methods is:']);
sum(time_array, 2)/runs


disp(['The average iteration count of the various methods is:']);
sum(iters_array,2)/runs

disp(['The average reconstruction accuracy of the various methods is:']);
sum(is_equal,2)/runs






% plot results: of course, only the last run will be shown
figure;
handle = semilogy(1:length(conv_hist_alg_log), conv_hist_alg_log, 'k-*', ...
         1:length(conv_hist_alg_log_sylv), conv_hist_alg_log_sylv, 'k-s', ...
         1:length(conv_hist_alg_log_sylvC),conv_hist_alg_log_sylvC, 'k-o',...
         1:length(conv_hist_pS), conv_hist_pS, 'k:d',...
         1:length(conv_hist_pSu), conv_hist_pSu, 'k--.');
     
set(handle(1),'linewidth',1, 'MarkerSize', 5);
set(handle(2),'linewidth',1, 'MarkerSize', 5);
set(handle(3),'linewidth',1, 'MarkerSize', 5); 
set(handle(4),'linewidth',1, 'MarkerSize', 5);
set(handle(5),'linewidth',1, 'MarkerSize', 5);

legend(['alg log'],...
       ['alg log sylv'],...
       ['alg log sylv + Cayley'],...
       ['p shooting (2 t-steps)'],...
       ['p shooting (4 t-steps)'])
