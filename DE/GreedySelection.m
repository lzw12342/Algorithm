function [P, P_fit] = GreedySelection(P, P_fit, T, T_fit)
% Greedy selection for Differential Evolution (one-to-one survivor selection)
%
%   [P, P_fit] = GreedySelection(P, P_fit, T, T_fit)
%
% Inputs:
%   P      - Parent population, size (D × N)
%   P_fit  - Fitness of parents, size (1 × N)
%   T      - Trial vectors (after crossover), size (D × N)
%   T_fit  - Fitness of trial vectors, size (1 × N)
%
% Outputs:
%   P      - Updated population after selection
%   P_fit  - Updated fitness values
%
% Note: Assumes minimization problem (T replaces P if T_fit <= P_fit)

    % Input validation (optional but recommended)
    if ~isequal(size(P), size(T))
        error('P and T must have the same dimensions');
    end
    if length(P_fit) ~= size(P, 2) || length(T_fit) ~= size(T, 2)
        error('Fitness vectors must match population size');
    end
    
    % Greedy replacement: keep the better one
    improvement = T_fit <= P_fit;  % For minimization
    
    P(:, improvement) = T(:, improvement);
    P_fit(improvement) = T_fit(improvement);
end