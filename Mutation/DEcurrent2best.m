function V = DEcurrent2best(P, best, F)
% DE/current-to-best/1 mutation operator
% Formula: v_i = x_i + F*(x_best - x_i) + F*(x_r1 - x_r2)
%
% Inputs:
%   P    : (D x N) parent population matrix
%   best : (D x 1) best individual column vector
%   F    : scalar or (1 x N) scaling factor
%
% Output:
%   V    : (D x N) mutant vectors

    N = size(P, 2);
    
    % Generate random permutations for selecting distinct indices
    [~, rnd] = sort(rand(N, N), 1);   % (N x N)
    
    % Random indices for the differential component (x_r1 - x_r2)
    r1 = rnd(1, :);                   % (1 x N)
    r2 = rnd(2, :);                   % (1 x N)
    
    % Base vectors: current population itself (x_i)
    base = P;                         % (D x N)
    
    % Direction towards best individual (implicit broadcast: D x 1 minus D x N)
    best_dir = best - base;           % (D x N)
    
    % Differential vector between two random individuals
    diff_dir = P(:, r1) - P(:, r2);  % (D x N)
    
    % Assembly: x_i + F*(x_best - x_i) + F*(x_r1 - x_r2)
    V = base + F .* best_dir + F .* diff_dir;
end