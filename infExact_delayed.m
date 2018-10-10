function [nlZ dnlZ nlhood posterior_mean posterior_covariance] = infExact_delayed(hyp, mean, cov, lik, T,N,p,sign,x,y,DX,X0)

% Exact inference for a GP with Gaussian likelihood. Compute the negative log marginal likelihood and its derivatives
% w.r.t. the hyperparameters. See also "help infMethods".

if iscell(lik), likstr = lik{1}; else likstr = lik; end
if ~ischar(likstr), likstr = func2str(likstr); end
if ~strcmp(likstr,'likGauss')               % NOTE: no explicit call to likGauss
  error('Exact inference only possible with Gaussian likelihood');
end
 
n = size(y,1);
K = feval(cov{:}, hyp.cov, x);                      % evaluate covariance matrix
m = feval(mean{:}, hyp.mean, x);                          % evaluate mean vector, consant mean or zero mean
m_all=DX*ones(size(DX,2),1)*m(1);   % the mean of the marginal likelihood

%expand K and m
tmp1=(X0*X0').*K;
% K_total=kron(tmp1,eye(N));% simplify calculation of K_total=DX*K_extend*DX';
%%
%calculate the inverse of P
sn2 = exp(2*hyp.lik);                               % noise variance of likGauss
if sn2<1e-6                        % very tiny sn2 can lead to numerical trouble
    P=tmp1+sn2*eye(size(K,1));   sl =   1;
else
    P=tmp1/sn2+eye(size(K,1));   sl = sn2;
end

LL=chol(P);
L=kron(LL,eye(N));
LL_inv=LL\eye(size(LL,1));% LL_inv=inv(LL);
K_total_inv=kron(LL_inv*LL_inv',eye(N));
alpha=K_total_inv*(y-m_all)/sl;

if nargout>1                               % do we want the marginal likelihood?
  nlZ = (y-m_all)'*alpha/2 + sum(log(diag(L))) + n*log(2*pi*sl)/2;   % -log marg lik
  if nargout>=2                                         % do we want derivatives?
    dnlZ = hyp;                                 % allocate space for derivatives
    tmp0=kron(LL_inv,eye(N));
     Q = tmp0*tmp0'/sl - alpha*alpha';     % precompute for convenience
    for i = 1:numel(hyp.cov)
        dnlZ.cov(i) = sum(sum(Q.*(kron((X0*X0').*(feval(cov{:}, hyp.cov, x, [], i)),eye(N)))))/2;
    end
    dnlZ.lik = sn2*trace(Q);
    for i = 1:numel(hyp.mean), 
        tmp=feval(mean{:}, hyp.mean, x, i);
        dnlZ.mean(i) = -(DX*ones(size(DX,2),1)*tmp(1))'*alpha;
    end
  end
end
if(sign)
    K_extend=kron(K,eye(size(DX,2)/length(x)));  %kronecker product
    m_extend=ones(size(DX,2),1)*m(1);
    posterior_mean=m_extend+(DX*K_extend)'*alpha;
    posterior_covariance=K_extend-(DX*K_extend)'*(inv(L)*inv(L'))/sl*(DX*K_extend);
    nlhood= (y-m_all)'*(y-m_all)/exp(2*hyp.lik)/2+hyp.lik + n*log(2*pi)/2;
end