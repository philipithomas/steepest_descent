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

f = sym('5*x_1^2+x_2^2+4*x_1*x_2-14*x_1-6*x_2+20');
grad = gradient(f);
hessian = hessian(f);
syms x_1
syms x_2
threshhold = 10^(-6);


%% Set up initial conditions
k = 1 % iteration count;
x = [0 10]';
d = [-26 -14]';
alpha = ( d(1)^2 + d(2)^2 ) / (2 * ( 5 * d(1)^2 + d(2)^2 + 4*d(1)*d(2) ) );

results(k,:) = [k, x(1), x(2), d(1), d(2), norm(d), alpha, subs(f,[x_1 x_2], x') ]
x = x + alpha .* (d);
    
%% Run algorithm


while 1>0
    disp('---');
    x_prior = x;
    k = k+1;
    d = -1 * subs(grad,[x_1 x_2], x'); % find new direction
    alpha = ( d(1)^2 + d(2)^2 ) / (2 * ( 5 * d(1)^2 + d(2)^2 + 4*d(1)*d(2) ) );
    
    x = x + alpha .* (d);
    
    f_k = subs(f,[x_1 x_2], x');
    
    results(k,:) = [k, x_prior(1), x_prior(2), d(1), d(2), norm(d), alpha, subs(f,[x_1 x_2], x_prior') ];

    if norm(x - x_prior,2) < threshhold
        break;
    end
end
