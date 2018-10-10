
% example 2
clear all,clc,close all
addpath(genpath(pwd))

%% generate simulated data
T=500;
w = 0.7*sin([1:T]/100);
x = randn(1,T);
y = w.*x + 0.3*randn(1,T);
Data = [x',y'];
causal_ordering=[1,2]; % give hypothetical causal ordering

% only consider instantaneous causal relations
[B,p_val] = Tdepent_FCM_ins(Data, causal_ordering);
% INPUT:
%   Data : TxN matrix of samples(T: number of samples; N: number of variables)
%   causal_ordering:   1xN vector. The root node is labelled as 1, and the sink node is labelled as N
%       for example: if x1->x2->x3, and Data = [x1,x2,x3], then causal ordering = [1,2,3];
%                    if x3->x2->x1, and Data = [x1,x2,x3], then causal ordering = [3,2,1];

% OUTPUT:
%   B: the estimated posterior mean of the instantaneous causal coefficients
%      B(i,j,:): means the causal coefficicents from Xj to Xi (Xj -> Xi)
%   p_val: p values derived from the independence test between estimated noise term and hypothetical causes


% check the p values. If they are all lager than the significance level,
% then we accecpt the hypothetical instantaneous causal ordering; otherwise, try other
% possible causal orderings
