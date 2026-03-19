function [O1, O2] = LogX(P, I1, I2, s)
% LogX - Logistic Crossover (symmetric around the parents' midpoint)
%
% Inputs:
%   P     : D × N matrix, population
%   I1    : 1 × n vector, indices of first parents
%   I2    : 1 × n vector, indices of second parents
%   s     : scale parameter s (controls perturbation strength)
%           recommended values:
%             0.8 – 1.2     most common / balanced
%             1.0           default & usually good starting point
%             1.3 – 1.8     stronger exploration (when stuck in local optima)
%             0.5 – 0.8     more exploitation / finer search (when already close to good region)
%
% Outputs:
%   O1, O2 : two offspring matrices (D × n)

    if nargin < 4 || isempty(s)
        s = 1.0;           % most papers / implementations use 1.0
    end
    
    D = size(P, 1);
    I1 = I1(:)';  
    I2 = I2(:)';  
    n  = length(I1);
    
    if length(I2) ~= n
        error('I1 and I2 must have the same length');
    end
    
    % Generate logistic-distributed perturbation factors
    u = rand(D, n);
    beta = -s * log((1 - u) ./ u);   % logistic variate (mean=0, scale=s)
    
    p1   = P(:, I1);
    p2   = P(:, I2);
    avg  = 0.5 * (p1 + p2);
    delta = abs(p1 - p2);
    
    % Two symmetric offspring
    O1 = avg + beta .* delta;
    O2 = avg - beta .* delta;
end
