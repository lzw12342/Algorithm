function T = BinomialCrossover(P, V, CR)
% Binomial Crossover (Binomial Recombination) for Differential Evolution
% Implements the standard binomial crossover operator used in DE.
%
%   T = BinomialCrossover(P, V, CR)
%
% Inputs:
%   P     - Parent (target) population matrix, size (D × N)
%           where D is the problem dimension, N is the population size
%   V     - Mutant (donor) vectors, size (D × N)
%   CR    - Crossover probability (scalar in [0,1])
%
% Output:
%   T     - Trial vectors after crossover, size (D × N)
%
% Description:
%   For each dimension j and each individual i:
%     - With probability CR, inherit from the mutant V(:,i)
%     - With probability (1-CR), inherit from the parent P(:,i)
%   To guarantee at least one component from the mutant (avoiding degenerate cases),
%   a random position j_rand(i) is forced to come from V for each individual i.
%
%   This is the classic binomial crossover used in DE/rand/1/bin, DE/best/1/bin, etc.

    % Get dimensions
    [D, N] = size(P);

    % Generate uniform random numbers in [0,1] for each gene of each individual
    mask = rand(D, N) <= CR;          % true: take from mutant V, false: keep parent P

    % Enforce at least one gene from mutant (standard DE j_rand mechanism)
    % Select one random dimension per individual and force crossover there
    j_rand = randi(D, 1, N);          % random row indices (1 to D) for each column
    mask(sub2ind([D, N], j_rand, 1:N)) = true;

    % Perform binomial crossover (vectorized)
    T = P;                            % start with parent
    T(mask) = V(mask);                % replace selected positions with mutant values

end
