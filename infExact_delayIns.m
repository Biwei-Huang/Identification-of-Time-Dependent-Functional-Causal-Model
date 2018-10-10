function [nlZ dnlZ posterior_mean posterior_covariance] = infExact_delayIns(hyp, mean, cov, lik, T,N,p,sign,x, y,DX)

% Exact inference for a GP with Gaussian likelihood. Compute a parametrization
% of the posterior, the negative log marginal likelihood and its derivatives
% w.r.t. the hyperparameters. See also "help infMethods".
%
% update the code from Carl Edward Rasmussen and Hannes Nickisch, 2014-03-04.


if iscell(lik), likstr = lik{1}; else likstr = lik; end
if ~ischar(likstr), likstr = func2str(likstr); end
if ~strcmp(likstr,'likGauss')               % NOTE: no explicit call to likGauss
  error('Exact inference only possible with Gaussian likelihood');
end
 
T_train=length(x);
n = size(y,1);
K = feval(cov{:}, hyp.cov, x);                      % evaluate covariance matrix
m = feval(mean{:}, hyp.mean, x);                          % evaluate mean vector
m_extend=ones((N*(N*p+1)+N*(N-1)/2)*T_train,1)*m(1);
m_all=DX*m_extend;   % the mean of the marginal likelihood

K_extend=kron(K,eye(N*(N*p+1)+N*(N-1)/2));  %kronecker product
K_total=DX*K_extend*DX';
sn2 = exp(2*hyp.lik);                               % noise variance of likGauss
if sn2<1e-6                        % very tiny sn2 can lead to numerical trouble
  L = chol(K_total+sn2*eye(n)); sl =   1;   % Cholesky factor of covariance with noise
  pL = -solve_chol(L,eye(n));                            % L = -inv(K+inv(sW^2))
else
  L = chol(K_total/sn2+eye(n)); sl = sn2;                       % Cholesky factor of B
  pL = L;                                           % L = chol(eye(n)+sW*sW'.*K)
end

alpha = solve_chol(L,y-m_all)/sl;

if nargout>1                               % do we want the marginal likelihood?
  nlZ = (y-m_all)'*alpha/2 + sum(log(diag(L))) + n*log(2*pi*sl)/2;   % -log marg lik
  if nargout>=2                                         % do we want derivatives?
    dnlZ = hyp;                                 % allocate space for derivatives
    Q = solve_chol(L,eye(n))/sl - alpha*alpha';     % precompute for convenience
    for i = 1:numel(hyp.cov)
        dnlZ.cov(i) = sum(sum(Q.*(DX*kron(feval(cov{:}, hyp.cov, x, [], i),eye(N*(N*p+1)+N*(N-1)/2))*DX')))/2;
    end
    dnlZ.lik = sn2*trace(Q);
    for i = 1:numel(hyp.mean), 
        tmp=feval(mean{:}, hyp.mean, x, i);
        dnlZ.mean(i) = -(DX*ones((N*(N*p+1)+N*(N-1)/2)*T_train,1)*tmp(1))'*alpha;
    end
  end
end
if(sign)
posterior_mean=m_extend+(DX*K_extend)'*alpha;
posterior_covariance=K_extend-(DX*K_extend)'*(inv(L)*inv(L'))*(DX*K_extend);
end
