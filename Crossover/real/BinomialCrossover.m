function T = BinomialCrossover(P, V, pc)
% Binomial Crossover (Binomial Recombination) for Differential Evolution
% Implements the standard binomial crossover operator used in DE.
%
%   T = BinomialCrossover(P, V, CR)
%
% Inputs:
%   P     - Parent (target) population matrix, size (D × N)
%           where D is the problem dimension, N is the population size
%   V     - Mutant (donor) vectors, size (D × N)
%   CR    - Crossover probability, either:
%             scalar  : same CR applied to all individuals
%             1 × N row vector : individual-specific CR for each member
%
% Output:
%   T     - Trial vectors after crossover, size (D × N)
%
% Description:
%   For each dimension j and each individual i:
%     - With probability CR(i), inherit from the mutant V(:,i)
%     - With probability (1-CR(i)), inherit from the parent P(:,i)
%   To guarantee at least one component from the mutant (avoiding degenerate cases),
%   a random position j_rand(i) is forced to come from V for each individual i.
%
%   This is the classic binomial crossover used in DE/rand/1/bin, DE/best/1/bin, etc.

    % Get dimensions
    [D, N] = size(P);

    % Validate and broadcast CR to (1 × N)
    if isscalar(pc)
        pc = repmat(pc, 1, N);        % scalar → uniform row vector
    elseif ~isequal(size(pc), [1, N])
        error('CR must be a scalar or a 1×N row vector (got size [%s]).', ...
              num2str(size(pc)));
    end

    % Generate uniform random numbers in [0,1] for each gene of each individual
    % pc is (1×N), rand(D,N) is (D×N) → broadcasting compares each column with its own CR
    mask = rand(D, N) <= pc;          % true: take from mutant V, false: keep parent P

    % Enforce at least one gene from mutant (standard DE j_rand mechanism)
    % Select one random dimension per individual and force crossover there
    j_rand = randi(D, 1, N);          % random row indices (1 to D) for each column
    mask(sub2ind([D, N], j_rand, 1:N)) = true;

    % Perform binomial crossover (vectorized)
    T = P;                            % start with parent
    T(mask) = V(mask);                % replace selected positions with mutant values
end