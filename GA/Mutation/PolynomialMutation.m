function P = PolynomialMutation(P, lb, ub, scale)
% PolynomialMutation - Vectorized Polynomial Mutation (Deb's standard real-coded mutation)
%   Applies the polynomial probability distribution mutation to decision variables.
%   This is the exact formulation used in NSGA-II and many real-coded genetic algorithms.
%
%   References:
%     K. Deb & S. Agrawal (1995). Simulated binary crossover and polynomial mutation
%     for real-coded genetic algorithms. (Standard polynomial mutation operator)
%
% Inputs:
%   P     : D × N matrix     – current population (each column is one individual)
%   lb    : D × 1 vector or scalar – lower bounds for each dimension
%   ub    : D × 1 vector or scalar – upper bounds for each dimension
%   scale : scalar           – distribution index η_m (mutation strength parameter)
%                             Higher values → smaller perturbations near parent
%                             Typical values: 20 (common default in NSGA-II literature)
%
% Output:
%   P     : D × N matrix     – mutated population (in-place modification)

    if nargin < 4 || isempty(scale)
        scale = 20;           % Standard default distribution index from Deb (1995/2001)
    end

    [D, N] = size(P);

    % Step 1: Generate mutation mask – each gene mutates independently with probability 1/D
    %         (Note: original MR was removed per request; using classic 1/D rate)
    mutate_prob = 1 / D;                  % Standard choice in Deb's polynomial mutation
    mutate_mask = rand(D, N) < mutate_prob;

    % Early return if no genes are selected for mutation
    idx_flat = find(mutate_mask);
    if isempty(idx_flat)
        return;
    end

    % Step 2: Extract bounds corresponding to mutated positions (vectorized)
    rows = mod(idx_flat - 1, D) + 1;      % row indices of mutated elements
    xi   = P(idx_flat);                   % current values at mutated positions
    lbi  = lb(rows);                      % lower bounds at those positions
    ubi  = ub(rows);                      % upper bounds at those positions
    range_i = ubi - lbi;

    % Step 3: Normalize distances to boundaries
    d1 = (xi - lbi) ./ range_i;           % normalized distance to lower bound [0,1]
    d2 = (ubi - xi) ./ range_i;           % normalized distance to upper bound [0,1]

    % Precompute exponents (used in both branches)
    nm     = scale + 1;
    inv_nm = 1 / nm;

    % Step 4: Generate perturbation δq using polynomial distribution
    u  = rand(size(xi));                  % uniform random [0,1] for each mutated gene
    dq = zeros(size(xi));

    % Branch A: perturbation towards lower bound (u ≤ 0.5)
    mask_l = u <= 0.5;
    if any(mask_l)
        u_l = u(mask_l);
        % Exact polynomial formula (Deb 1995):
        % δ_q = (2u + (1-2u)(1-d1)^{η_m+1})^{1/(η_m+1)} - 1
        val = 2 * u_l + (1 - 2 * u_l) .* (1 - d1(mask_l)).^nm;
        dq(mask_l) = val.^inv_nm - 1;
    end

    % Branch B: perturbation towards upper bound (u > 0.5)
    mask_r = ~mask_l;
    if any(mask_r)
        u_r = u(mask_r);
        % Exact polynomial formula (Deb 1995):
        % δ_q = 1 - (2(1-u) + 2(u-0.5)(1-d2)^{η_m+1})^{1/(η_m+1)}
        val = 2 * (1 - u_r) + 2 * (u_r - 0.5) .* (1 - d2(mask_r)).^nm;
        dq(mask_r) = 1 - val.^inv_nm;
    end

    % Step 5: Apply perturbation and clamp to bounds (handles floating-point issues)
    new_values = xi + dq .* range_i;
    P(idx_flat) = max(lbi, min(ubi, new_values));
end