function K = covLapla(hyp, x, z, i)

% Laplacian Kernel
%  exp(-|x-y|/ell)/(sf^2)
% where the P matrix is ell^2 times the unit matrix and sf^2 is the signal
% variance. The hyperparameters are:
%
% hyp = [ log(ell)
%         log(sf)  ]
%
% For more help on design of covariance functions, try "help covFunctions".
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-09-10.
%
% See also COVFUNCTIONS.M.

if nargin<2, K = '2'; return; end                  % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = numel(z)==0; dg = strcmp(z,'diag') && numel(z)>0;        % determine mode

ell = exp(hyp(1));                                 % characteristic length scale
sf2 = exp(2*hyp(2));                                           % signal variance

% precompute squared distances
if dg                                                               % vector kxx
  K = zeros(size(x,1),1);
else
  if xeqz                                                 % symmetric matrix Kxx
    K = squart(sq_dist(x'/ell));
  else                                                   % cross covariances Kxz
    K = squart(sq_dist(x'/ell,z'/ell));
  end
end

if nargin<4                                                        % covariances
  K = sf2*exp(-K);
else                                                               % derivatives
  if i==1
    K = sf2*exp(-K).*K;     % derivative of ell
  elseif i==2
    K = 2*sf2*exp(-K);      % derivative of sf
  else
    error('Unknown hyperparameter')
  end
end