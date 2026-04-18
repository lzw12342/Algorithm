function P = ReInitializationBoundary(P, lb, ub)
% Re-initialization Boundary Handling (Vectorized Implementation)
% Resets out-of-bound particles to random positions within bounds.
%
% Input:
%   P  - D×N matrix (D-dimensional decision variables, N individuals/particles)
%   lb - D×1 column vector (lower bounds for each dimension) or scalar
%   ub - D×1 column vector (upper bounds for each dimension) or scalar
%
% Output:
%   P  - D×N matrix with out-of-bound positions randomly re-initialized within [lb, ub]

    lb = lb(:);
    ub = ub(:);

    % Identify all out-of-bound positions (below lower bound or above upper bound)
    mask_out = (P < lb) | (P > ub);

    if any(mask_out, 'all')
        % Get row indices of out-of-bound elements (i.e., dimension indices d)
        [rows, ~] = find(mask_out);  % rows is an n_out×1 vector
        % Generate random values within [lb(d), ub(d)] for each violated position
        % lb(rows) and ub(rows) automatically extract bounds for corresponding dimensions
        rand_vals = lb(rows) + rand(length(rows), 1) .* (ub(rows) - lb(rows));
        % Assign re-initialized values back to out-of-bound positions
        P(mask_out) = rand_vals;
    end
end
