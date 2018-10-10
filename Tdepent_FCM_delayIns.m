
function [A, G, B, p_val] = Tdepent_FCM_delayIns(Data, causal_ordering,p)

% model type:
%  linear model (equation 2 in the paper), and consider time-delayed and
%  instantaneous causal effects simultaneously. We estimate them in one
%  step.

% In this code we assume all time-dependent coefficients share the same
% kernel hypermeters

% INPUT:
%   Data : TxN matrix of samples(T: number of samples; N: number of variables)
%   p: time lag
%   causal_ordering:  the instantaneous causal ordering. 1xN vector. 
%       The root node is labelled as 1, and the sink node is labelled as N
%       for example: if x1->x2->x3, and Data = [x1,x2,x3], then causal ordering = [1,2,3];
%                    if x3->x2->x1, and Data = [x1,x2,x3], then causal ordering = [3,2,1];


% OUTPUT:
%   A: the estimated posterior mean of time-delayed causal coefficients
%     A{i}(j,k,:): the ith time-lagged causal coefficients from Xk to Xj(Xk ->Xj)
%   G: the estimated posterior mean of confounder term
%     G(i,:): the confounder term for Xi
%   B: the estimated posterior mean of the instantaneous causal coefficients
%      B(i,j,:): means the causal coefficicents from Xj to Xi (Xj -> Xi)
%   p_val: p values derived from the independence test between estimated noise term and hypothetical causes

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
for i=1:size(time_series,1)
    time_series(i,:)=time_series(i,:)-mean(time_series(i,:)); % should subtract the mean information
    time_series(i,:)=time_series(i,:)/std(time_series(i,:));
end


% settings of the kernel
meanfunc = {@meanZero};
hyp.mean=[];
likfunc=@likGauss;
sn=0.1;  %standard deviation of noise
hyp.lik=log(sn);
covfunc = {@covSEiso};
ell_1=1.1; sf=1.1;
hyp.cov = [log(ell_1);log(sf)];

% reordering the data according to the causal ordering
time_series = time_series(causal_ordering,:);

%%
train_t = [p+1:T]'; % time-delayed part
T_train = length(train_t); 
train_x_delayed = [];
for i=p:-1:1
    train_x_delayed=[train_x_delayed,time_series(:,train_t-i)'];
end
tmp=ones(size(train_x_delayed,1),1);
train_x_delayed=[train_x_delayed,tmp];

train_y=time_series(:,train_t)';
train_x_ins = train_y;
train_yv=(reshape(train_y',size(train_y,1)*size(train_y,2),1)); %reshape it to a column vector

DX_train=zeros(N*T_train,(N*(N*p+1)+N*(N-1)/2)*T_train);
for i = 1:T_train
    for j=1:N
        DX_train((i-1)*N+j,((i-1)*(N*(N*p+1)+N*(N-1)/2))+(j-1)*(N*p+1)+1:((i-1)*(N*(N*p+1)+N*(N-1)/2))+j*(N*p+1))=train_x_delayed(ceil(((i-1)*N+j)/N),:);
        DX_train((i-1)*N+j,((i-1)*(N*(N*p+1))+j*(N*p+1)+(i-1)*N*(N-1)/2)+(j-2)*(j-1)/2+1:((i-1)*(N*(N*p+1))+j*(N*p+1)+(i-1)*N*(N-1)/2)+(j-2)*(j-1)/2+(j-1)) = train_x_ins(i,1:j-1);
    end
end

[hyp2,fhyp2] = minimize(hyp, @gp_delayIns, 1000, @infExact_delayIns, meanfunc, covfunc, likfunc,T,N,p,0, train_t,train_yv,DX_train);
[nlZ dnlZ posterior_mean posterior_covariance]=infExact_delayIns(hyp2, meanfunc, covfunc, likfunc,T,N,p,1, train_t,train_yv,DX_train);

p_mean=reshape(posterior_mean, N*(N*p+1)+N*(N-1)/2, length(posterior_mean)/(N*(N*p+1)+N*(N-1)/2));%every row is the mean of one gaussian process with time
posterior_variance=diag(posterior_covariance);
p_variance=reshape(posterior_variance, N*(N*p+1)+N*(N-1)/2, length(posterior_variance)/(N*(N*p+1)+N*(N-1)/2));
number_of_func=N*(N*p+1)+N*(N-1)/2;

% estimated posterior mean of time-delayed coefficients and coufounder terms
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

%plot the time-depedent delayed coefficients
figure
for i=1:N
    z=[p+1:T];
    for j=1:N*p+1
        subplot(N,N*p+1,(i-1)*(N*p+1)+j);
        plot(z,p_mean((i-1)*(N*p+1)+j,:)','r');
        title('time-dependent delayed causal coefficients');
    end
end


% estimated posterior mean of instantaneous coefficients
B = zeros(N,N,T-p);
for i=1:N
    for j=1:N
        if(j<=i-1)
            B(i,j,:) = p_mean(N*(N*p+1)+(i-2)*(i-1)/2+j,:);
        end
    end
end

% plot the time-depedent instantaneous coefficients
figure
for i=1:N-1
    z=[p+1:T]';
    for j=1:N-1
        subplot(N-1,N-1,(i-1)*(N-1)+j);
        plot(z,p_mean(N*(N*p+1)+i*(i-1)/2+j,:)','r');
        title('time-dependent instantaneous causal coefficients');
    end
end


% test whether the estimated noise term is independent of hypothetical causes
noise_vector=train_yv-DX_train*posterior_mean;
noise = (reshape(noise_vector,N,size(noise_vector,1)/N))';
for i=1:size(noise,2) % normalization
    noise(:,i) = noise(:,i)-mean(noise(:,i));
    noise(:,i) = noise(:,i)/std(noise(:,i));
end

count = 0;
for i=2:N
    for j=1:N-1   % should be independent
        count = count+1;
        p_val(count)=UInd_KCItest(train_x_ins(:,j),noise(:,i),[]);
    end
end
% check the p values. If they are all lager than the significance level,
% then we accecpt the hypothetical instantaneous causal ordering; otherwise, try other
% possible causal orderings


















