%% Intro
% This function is an implementation of the steepest descent optimization
% algorithm for the function below defined as 'f'. 

%% Clear environment
clear
clc
format long

%% Initialize results table
results =zeros(1,8);
%Guide to results table:
%Columns, in order, are:
% k, x1_k, x2_k, d1_k, d2_k, norm(d), alpha_k, f(x_k)

%% Initialize variables

f = sym('10*a^2 + 10*a*b + b^2 - 14*a -6*b + 15 ');
grad = gradient(f);
hessian = hessian(f);
syms a
syms b
threshhold = 10^(-6);


%% Set up initial conditions
k = 0; % iteration count;
x = [40 -100]';
    
%% Run algorithm


while 1>0
    x_prior = x;
    k = k+1;
    d = -1 * subs(grad,[a b], x'); % find new direction
    alpha = ( d(1) * (14 - 20*x(1) -10 * x(2)) +d(2)*( 6 - 10*x(1) - 2*x(2)) ) / ( 20*d(1)^2 + 20*d(1)*d(2) + 2*d(2)^2);
    
    x = x + alpha .* (d);
    
    f_k = subs(f,[a b], x');
    
    results(k,:) = [k, x_prior(1), x_prior(2), d(1), d(2), norm(d), alpha, subs(f,[a b], x_prior') ];

    if norm(x - x_prior,2) < threshhold
        break;
    end
end
