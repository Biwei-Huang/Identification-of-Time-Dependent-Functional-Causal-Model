
% example 2
clear all,clc,close all
addpath(genpath(pwd))

%% generate simulated data
T=500;
x(1)=0.1*randn;y(1)=0.*randn;
for t=2:T
    x(t)=0.6*x(t-1)+0.35*y(t-1)+0.3*randn;
    y(t)=0.5*y(t-1)+ 0.4*x(t)+0.3*randn;
end
Data = [x',y'];
p=1;
causal_ordering = [1,2];

% only consider instantaneous causal relations
[A, G, B, p_val] = Tdepent_FCM_delayIns(Data, causal_ordering,p);

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

% check the p values. If they are all lager than the significance level,
% then we accecpt the hypothetical instantaneous causal ordering; otherwise, try other
% possible causal orderings