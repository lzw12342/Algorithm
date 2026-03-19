function T = DErand1(P, I1, I2, I3, F)
% Differential Evolution mutation operator: DE/rand/1 variant
% Uses externally provided index vectors for the three parents.
%
%   T = DErand1(P, I1, I2, I3, F)
%
% Inputs:
%   P     - Parent population matrix, size (D × N)
%   I1    - Row vector (1×n), indices for base vector
%   I2    - Row vector (1×n), indices for first difference vector
%   I3    - Row vector (1×n), indices for second difference vector
%   F     - Scaling factor (scalar or 1×n vector)
%
% Output:
%   T     - Mutant vectors, size (D × n)

    T = P(:, I1) + F .* (P(:, I2) - P(:, I3));   % (D × n)

end
