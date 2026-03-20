function I = RouletteWheelSelection(P_fit, sz1, sz2)
% RouletteWheelSelection - Optimized Roulette wheel selection (minimization)
%
% Inputs:
%   P_fit : (1 x N) fitness values (minimization problem)
%   sz1   : number of rows in output
%   sz2   : number of columns in output
%
% Output:
%   I     : (sz1 x sz2) selected indices

    N = numel(P_fit);
    if nargin < 2 || isempty(sz1), sz1 = 1; end
    if nargin < 3 || isempty(sz2), sz2 = N; end

    % 1. Convert minimization fitness to selection probability.
    % We use (max - fit) or 1/fit. For 1/fit, adding eps prevents division by zero.
    f_inv = 1 ./ (P_fit + eps);
    prob  = f_inv / sum(f_inv);
    
    % 2. Create the cumulative distribution (0 to 1)
    cum = [0, cumsum(prob)];
    % Ensure the last element is exactly 1 to avoid precision issues
    cum(end) = 1; 

    % 3. Generate all random spins at once
    r = rand(1, sz1 * sz2);

    % 4. Vectorized Search: histcounts is much faster than find in a loop.
    % It bins the random numbers 'r' into the edges defined by 'cum'.
    [~, ~, I_linear] = histcounts(r, cum);

    % 5. Reshape to requested dimensions
    I = reshape(I_linear, sz1, sz2);
end