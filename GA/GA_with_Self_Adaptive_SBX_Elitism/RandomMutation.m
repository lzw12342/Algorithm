function P = RandomMutation(P)
% RandomMutation - Vectorized Random Mutation based on Fanggidae et al. (2024)
% 
% This version mutates exactly ONE random gene for EVERY individual in the 
% population (D x N). It uses linear indexing to avoid loops.
%
% Formula: g' = g + r * (g_max - g_min), where r ~ Uniform[-0.1, 0.1]
%
% Inputs:
%   P : (D x N) matrix - Current population
%
% Output:
%   P : (D x N) matrix - Mutated population

    [D, N] = size(P);

    % 1. Compute per-dimension range across the population (D x 1)
    g_max = max(P, [], 2);
    g_min = min(P, [], 2);
    g_range = g_max - g_min;

    % 2. Pick exactly one random gene index (row) for each individual (column)
    % row_indices will be (1 x N)
    row_indices = randi(D, 1, N);
    
    % 3. Convert (row, col) coordinates to Linear Indices for fast access
    % Linear Index = row + (col - 1) * total_rows
    col_indices = 1:N;
    linear_indices = row_indices + (col_indices - 1) * D;

    % 4. Generate perturbation r ~ Uniform[-0.1, 0.1] for all individuals
    r = -0.1 + 0.2 * rand(1, N);

    % 5. Apply mutation using vectorized indexing
    % We index into the range vector using the selected row_indices
    P(linear_indices) = P(linear_indices) + r .* g_range(row_indices)';

end