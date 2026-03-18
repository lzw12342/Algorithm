function [O1, O2] = LogX(P, PI1, PI2, eta)
% LogX Logistic Crossover (LogX)
% Input:
%   P    : D×N population matrix (real-valued or integer encoding)
%   PI1  : 1×n or n×1 vector, indices of first parents
%   PI2  : 1×n or n×1 vector, indices of second parents
%   eta  : scale parameter s (default = 1.0)
% Output:
%   O1   : D×n offspring 1  (avg + β·|p1-p2|)
%   O2   : D×n offspring 2  (avg - β·|p1-p2|)

    if nargin < 4 || isempty(eta)
        eta = 1.0;
    end
    
    [D, ~] = size(P);
    
    % Force row vectors
    PI1 = PI1(:)';  
    PI2 = PI2(:)';  
    n = length(PI1);
    
    if length(PI2) ~= n
        error('PI1 and PI2 must have the same length');
    end
    
    % Generate D×n uniform random numbers [0,1]
    u = rand(D, n);
    
    % Logistic random variate: β = -s · log((1-u)/u)
    % (location μ=0, scale s=eta)
    beta = -eta * log((1 - u) ./ u);
    
    % Extract parents
    p1 = P(:, PI1);   % D × n
    p2 = P(:, PI2);   % D × n
    
    % Center and absolute difference
    avg   = 0.5 * (p1 + p2);
    delta = abs(p1 - p2);
    
    % Two complementary offspring
    O1 = avg + beta .* delta;   % η style (positive perturbation)
    O2 = avg - beta .* delta;   % ξ style (negative perturbation)
end
