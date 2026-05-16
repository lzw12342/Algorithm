function T = BinomialCrossover(P, V, pc)
% Binomial Crossover (Binomial Recombination) for Differential Evolution
% Implements the standard binomial crossover operator used in DE.
%
%   T = BinomialCrossover(P, V, pc)
%
% Inputs:
%   P   - Parent (target) population matrix, size (D × N)
%           where D is the problem dimension, N is the population size
%   V   - Mutant (donor) vectors, size (D × N)
%   pc  - Crossover probability, either:
%             scalar  : same CR applied to all individuals, broadcasts naturally
%             1 × N   : individual-specific CR, expanded to D × N before comparison
%
% Output:
%   T   - Trial vectors after crossover, size (D × N)
%
% Description:
%   For each dimension j and each individual i:
%     - With probability pc(i), inherit from the mutant V(:,i)
%     - With probability 1-pc(i), inherit from the parent P(:,i)
%   A random position j_rand(i) is forced to come from V for each individual i,
%   guaranteeing at least one component from the mutant (standard DE j_rand mechanism).

    [D, N] = size(P);

    % Validate and prepare pc
    if isequal(size(pc), [1, N])
        pc = pc(ones(D,1), :);        % 1×N → D×N, align for element-wise comparison
    elseif ~isscalar(pc)
        error('pc must be a scalar or a 1×N row vector (got size [%s]).', ...
              num2str(size(pc)));
    end

    % Generate crossover mask: true → take from mutant V
    mask = rand(D, N) <= pc;

    % Force at least one gene from mutant per individual (j_rand mechanism)
    j_rand = randi(D, 1, N);
    mask(sub2ind([D, N], j_rand, 1:N)) = true;

    % Assemble trial vectors
    T       = P;
    T(mask) = V(mask);
end