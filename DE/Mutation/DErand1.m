function T = DErand1(P, F)
% Differential Evolution mutation operator: DE/rand/1 (vectorized, strict distinct indices)
%
%   T = DErand1(P, F)
%
% Inputs:
%   P     - Parent population matrix, size (D × N), where D is dimensionality,
%           N is population size
%   F     - Scaling factor (scalar or 1×N vector)
%
% Output:
%   T     - Mutant (trial) vectors, size (D × N)
%
% Description:
%   Implements the classic DE/rand/1 mutation strategy:
%       v_i = x_{r1} + F_i × (x_{r2} - x_{r3})
%   where r1, r2, r3 are three **mutually distinct** random indices chosen
%   from 1 to N for each individual i.
%   Vectorized implementation using random permutations per column.
%   Safe for small populations (N ≥ 3 required for meaningful operation).

    % Get population size
    [D, N] = size(P);

    % Input validation (optional but recommended for strict version)
    if N < 3
        error('DE/rand/1 requires at least 3 individuals (N >= 3)');
    end

    % Generate random permutations for each of the N target vectors
    % rnd: (N × N) matrix, each column is a random permutation of 1:N
    [~, rnd] = sort(rand(N, N), 1);

    % For each column (each target individual), select three distinct random indices
    % We take the first three rows of the permutation matrix
    r1 = rnd(1, :);   % 1×N
    r2 = rnd(2, :);
    r3 = rnd(3, :);

    % Compute mutants using broadcasting
    % T(:,k) = P(:,r1(k)) + F(k) * (P(:,r2(k)) - P(:,r3(k)))
    T = P(:, r1) + F .* (P(:, r2) - P(:, r3));   % (D × N)

end
