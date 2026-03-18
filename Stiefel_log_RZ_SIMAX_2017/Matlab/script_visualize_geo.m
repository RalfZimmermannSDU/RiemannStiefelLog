%
% This script shows the geodesic path from
% a Stiefel matrix U (dim 1024x15) to the 
% canonical Stiefel point
% |I|
% |0| = E
%
% The data matrix stems from a proper orthogonal
% decomposition (POD) of state vectors for the 
% FitzHugh-Nagumo PDE
%

% load data
load_struct = load('U_FitzHugh.mat');
U = load_struct.Upod; 
[n,p] = size(U);

% canonical Stiefel center
E = [eye(p); zeros(n-p,p)];

% Compute tangent vector that sends U to E
tau = 1.0e-13;
[Delta, conv_hist] = Stiefel_Log(U, E, tau, 0);

% compute geodesic connecting U and E
res      = 101; % number of time steps
unit_int = linspace(0,1,res);

mat_list        = zeros(res, n,p);
mat_list(1,:,:) = U;

for k=2:res
    t = unit_int(k);
    mat_list(k,:,:) = Stiefel_Exp(U, t*Delta);
end

% did we end at the rigth spot?
Uend = reshape(mat_list(res,:,:), n,p);
if norm(E-Uend, 'fro') < 1.0e-11
    disp("Success: Geodesic reached its destination.")
else
    disp("Failure: Geodesic did not reach its destination.")
end

write_video = 1;

if write_video
    for k = 1:res
        figure("Visible","off")
        Ut = reshape(mat_list(k,:,:), n,p);
        surf(Ut, 'EdgeColor','none');
        view(135, 45)        % change camera angle
        axis([1 p 1 n -1 1]) % fix axes
        F(k) = getframe(gcf);% get current frame
        %drawnow
    end
    % many invisible figures are open
    % close them all
    close all;
    % create the video writer with 1 fps
    writerObj = VideoWriter(['Stiefel_geo','.avi']);
    writerObj.FrameRate = 25;   % set the seconds per image
    % open the video writer
    open(writerObj);
    % write the frames to the video
    for i=1:length(F)
        % convert the image to a frame
        frame = F(i) ;    
        writeVideo(writerObj, frame);
    end
    % close the writer object
    close(writerObj);
end % write video
