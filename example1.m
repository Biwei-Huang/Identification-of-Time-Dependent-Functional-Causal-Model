
% example 1
clear all,clc,close all
addpath(genpath(pwd))

%% generate simulated data
T=1000;
x(1)=0.01*randn;y(1)=0.01*randn;
for t=2:T
    x(t)=0.6*x(t-1) + 0.45*y(t-1)+0.1*randn;
    y(t)=0.4*x(t-1)+0.55*y(t-1)+0.1*randn;
end
Data = [x',y'];
p=1; % time lag

% only consider the delayed causal relations
[A G p_val] = Tdepent_FCM_delayed(Data, p);

% INPUT:
%   Data : TxN matrix of samples(T: number of samples; N: number of variables)
%   p: time lag

% OUTPUT:
%   A: the estimated posterior mean of time-delayed causal coefficients
%     A{i}(j,k,:): the ith time-lagged causal coefficients from Xk to Xj(Xk ->Xj)
%   G: the estimated posterior mean of confounder term
%     G(i,:): the confounder term for Xi
%   p_val: p values derived from the independence test between estimated noise terms
