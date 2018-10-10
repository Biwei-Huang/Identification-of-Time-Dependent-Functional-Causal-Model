Identification of time-dependent functional causal model

Copyright (c) 2015 Biwei Huang


### IMPORTANT FUNCTIONS

Recovering the time-delayed varying causal coefficients:

>> function [A G p_val] = Tdepent_FCM_delayed(Data, p)

* model type: 
   *  linear model (equation 5 in the paper), and only consider time-delayed causal effects

* in this code we assume all time-dependent coefficients share the same kernel width
* we apply a trick to make the computation much more efficient

* INPUT:
  * Data : TxN matrix of samples(T: number of samples; N: number of variables)
  * p: time lag

* OUTPUT:
  * A: the estimated posterior mean of time-delayed causal coefficients
    *    A{i}(j,k,:): the ith time-lagged causal coefficients from Xk to Xj(Xk ->Xj)
  * G: the estimated posterior mean of confounder term
    *    G(i,:): the confounder term for Xi
  *  p_val: p values derived from the independence test between estimated noise terms


### EXAMPLE:

see example1.m




Recovering the instantaneous varying causal coefficients:

>> function [B,p_val] = Tdepent_FCM_ins(Data,causal_ordering)

* model type: linear model, and only consider the time-dependent instantaneous causal effect. 
* the hypothetcal causal ordering needs to be assigned in advance
* in this code we assume all time-dependent coefficients share the same kernel width
* we apply a trick to make the computation much more efficient

* INPUT:
  * Data : TxN matrix of samples(T: number of samples; N: number of variables)
  * causal_ordering:   1xN vector. The root node is labelled as 1, and the sink node is labelled as N
     *  for example: if x1->x2->x3, and Data = [x1,x2,x3], then causal ordering = [1,2,3];
                    if x3->x2->x1, and Data = [x1,x2,x3], then causal ordering = [3,2,1];

* OUTPUT:
  *   B: the estimated posterior mean of the instantaneous causal coefficients
     *     B(i,j,:): means the causal coefficicents from Xj to Xi (Xj -> Xi)
  *  p_val: p values derived from the independence test between estimated noise term and hypothetical causes


### EXAMPLE

 see example2.m




Recovering both time-delayed and instantaneous varying causal coefficients, as well as cofounder terms:

>> function [A G p_val] = Tdepent_FCM_delayed(Data, p)

* model type:
 *  linear model (equation 2 in the paper), and consider time-delayed and instantaneous causal effects simultaneously. We estimate them in one step.

* In this code we assume all time-dependent coefficients share the same kernel hypermeters

* INPUT:
 *  Data : TxN matrix of samples(T: number of samples; N: number of variables)
 *   p: time lag
 *   causal_ordering:  the instantaneous causal ordering. 1xN vector. 
 *       The root node is labelled as 1, and the sink node is labelled as N
 *       for example: if x1->x2->x3, and Data = [x1,x2,x3], then causal ordering = [1,2,3];
 *                   if x3->x2->x1, and Data = [x1,x2,x3], then causal ordering = [3,2,1];


* OUTPUT:
 *   A: the estimated posterior mean of time-delayed causal coefficients
   *     A{i}(j,k,:): the ith time-lagged causal coefficients from Xk to Xj(Xk ->Xj)
 *   G: the estimated posterior mean of confounder term
   *     G(i,:): the confounder term for Xi
 *   B: the estimated posterior mean of the instantaneous causal coefficients
   *      B(i,j,:): means the causal coefficicents from Xj to Xi (Xj -> Xi)
 *  p_val: p values derived from the independence test between estimated noise term and hypothetical causes


### EXAMPLE:

see example3.m


### CITATION
	
Biwei Huang ,Kun Zhang , Bernhard Scholkopf, Causal discovery from nonstationary/heterogeneous data: skeleton estimation and orientation determination, IJCAI, 2015.