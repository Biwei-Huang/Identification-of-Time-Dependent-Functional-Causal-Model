
function [A G p_val] = Tdepent_FCM_delayed(Data, p)

% model type: 
   %  linear model (equation 5 in the paper), and only consider time-delayed causal effects

% in this code we assume all time-dependent coefficients share the same
% kernel width
% We apply a trick to make the computation much more efficient

% INPUT:
%   Data : TxN matrix of samples(T: number of samples; N: number of variables)
%   p: time lag

% OUTPUT:
%   A: the estimated posterior mean of time-delayed causal coefficients
%     A{i}(j,k,:): the ith time-lagged causal coefficients from Xk to Xj(Xk ->Xj)
%   G: the estimated posterior mean of confounder term
%     G(i,:): the confounder term for Xi
%   p_val: p values derived from the independence test between estimated noise terms

if (nargin <2)
    p = 1;  % set the default time lag of vector autoregression model
end

dpath=fullfile(pwd,'gpml-matlab-v3.4-2013-11-11','gpml-matlab-v3.4-2013-11-11');
addpath(dpath);
startup

time_series = Data';
T=size(time_series,2);  %number of time points
N=size(time_series,1);  %number of dimension of the data vector

% Data normalization
% for i=1:size(time_series,1)
%     time_series(i,:)=time_series(i,:)-mean(time_series(i,:)); % should subtract the mean information
%     time_series(i,:)=time_series(i,:)/std(time_series(i,:));
% end

% settings of the kernel
meanfunc = {@meanZero};
hyp.mean=[];
likfunc=@likGauss;
sn=0.1;  %standard deviation of noise
hyp.lik=log(sn);
covfunc = {@covSEiso};
ell_1=1.1; sf=1.1;
hyp.cov = [log(ell_1);log(sf)];

%%
train_t=[p+1:T]';
T_train=length(train_t); %the number of time point for training data

train_y=time_series(:,train_t)';
train_x=[];
for i=p:-1:1
    train_x=[train_x,time_series(:,train_t-i)'];
end
tmp=ones(size(train_x,1),1); 

train_x=[train_x,tmp];
train_yv=(reshape(train_y',size(train_y,1)*size(train_y,2),1)); %reshape it to a column vector

DX_train=zeros(N*T_train,N*(N*p+1)*T_train);
for i=1:N*T_train
    DX_train(i,(i-1)*(N*p+1)+1:i*(N*p+1))=train_x(ceil(i/N),:);
end

[hyp2,fhyp2] = minimize(hyp, @gp_my, 1000, @infExact_delayed, meanfunc, covfunc, likfunc,T,N,p,0, train_t,train_yv,DX_train,train_x);
[nlZ dnlZ nlhood posterior_mean posterior_covariance]=infExact_delayed(hyp2, meanfunc, covfunc, likfunc,T,N,p,1, train_t,train_yv,DX_train,train_x);

p_mean=reshape(posterior_mean, N*(N*p+1), length(posterior_mean)/(N*(N*p+1)));%every row is the posterior mean of one coefficient
posterior_variance=diag(posterior_covariance);
p_variance=reshape(posterior_variance, N*(N*p+1), length(posterior_variance)/(N*(N*p+1)));
number_of_func=N*(N*p+1);

% plot the time-depedent coefficient
figure
for i=1:number_of_func
    z=[p+1:T]';
    subplot(N,N*p+1,i)
    plot(z,p_mean(i,:)','r');
    title('time varying coefficients');
    hold off
end

A = cell(1,N); % lagged terms
G = []; % confounder terms
for i=p:-1:1 % lag
    for j=1:N
        for k = 1:N
        A{i}(j,k,:) = p_mean((j-1)*(N*p+1) + (p-i)*N+k,:);
        end
    end
end
      
for i=1:N
    G = [G;p_mean((i-1)*(N*p+1)+(N*p+1),:)];
end

% test the dependence between the estimated noise terms
noise_vector=train_yv-DX_train*posterior_mean;
noise=(reshape(noise_vector,N,size(noise_vector,1)/N))';

count=0;
for i=1:N-1
    for j=i+1:N   % should be independent
        count = count+1;
        p_val(count)=UInd_KCItest(noise(:,i), noise(:,j), []);        
    end
end

% check the p values. If the p values are all larger than significance
% level, then we do not need to a second step which considers the
% instantaneous causal effect; otherwise, a second step is necessary.

















