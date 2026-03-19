function V = DErand1(P, F)
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
%   V     - Mutant vectors, size (D × N)
%
% Description:
%   Implements the classic DE/rand/1 mutation strategy:
%       v_i = x_{r1} + F_i × (x_{r2} - x_{r3})
%   where r1, r2, r3 are three **mutually distinct** random indices chosen
%   from 1 to N for each individual i.
%   Vectorized implementation using random permutations per column.
%   Safe for small populations (N ≥ 3 required for meaningful operation).

    N = size(P,2);

    % Input validation (optional but recommended for strict version)
    if N < 3
        error('DE/rand/1 requires at least 3 individuals (N >= 3)');
    end

    [~, rnd] = sort(rand(N, N), 1);

    V = P(:, rnd(1, :)) + F .* (P(:, rnd(2, :)) - P(:, rnd(3, :)));   % (D × N)
end
