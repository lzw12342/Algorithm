function [O1, O2] = SBX(P, I1, I2, eta)
% SBX - Simulated Binary Crossover for real-valued variables
% Vectorized version (random numbers generated outside the loop)

    if nargin < 4 || isempty(eta)
        eta = 20;
    end
    
    D = size(P, 1);
    I1 = I1(:)';  
    I2 = I2(:)';  
    n  = length(I1);
    
    if length(I2) ~= n
        error('I1 and I2 must have the same length');
    end
    
    % Pre-generate all random numbers at once (D × n matrix)
    u = rand(D, n);
    
    % Decide which pairs perform crossover
    do_crossover = rand(1, n) < 0.9;   % or use fixed CR if preferred
    
    % Extract parents (D × n)
    p1 = P(:, I1);
    p2 = P(:, I2);
    
    % Initialize offspring
    O1 = p1;  % copy parents as default
    O2 = p2;
    
    % Only compute for pairs that actually crossover
    if ~any(do_crossover)
        return;
    end
    
    active = find(do_crossover);
    na = length(active);
    
    % Subset for active pairs
    u_active   = u(:, active);         % D × na
    p1_active  = p1(:, active);
    p2_active  = p2(:, active);
    
    % Compute beta (vectorized)
    beta = zeros(D, na);
    
    mask = u_active <= 0.5;
    beta(mask)  = (2 * u_active(mask)) .^ (1 / (eta + 1));
    
    mask_inv = ~mask;
    beta(mask_inv) = (1 ./ (2 * (1 - u_active(mask_inv)))) .^ (1 / (eta + 1));
    
    % Generate children (vectorized)
    child1 = 0.5 * ((1 + beta) .* p1_active + (1 - beta) .* p2_active);
    child2 = 0.5 * ((1 - beta) .* p1_active + (1 + beta) .* p2_active);
    
    % Write back to output
    O1(:, active) = child1;
    O2(:, active) = child2;
end
