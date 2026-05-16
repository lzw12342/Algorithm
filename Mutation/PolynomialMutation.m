function P = PolynomialMutation(P, lb, ub, pm, scale)
% PolynomialMutation - Vectorized Polynomial Mutation (Deb's standard real-coded mutation)
%   Applies the polynomial probability distribution mutation to decision variables.
%   This is the exact formulation used in NSGA-II and many real-coded genetic algorithms.
%
%   References:
%     K. Deb & S. Agrawal (1995). Simulated binary crossover and polynomial mutation
%     for real-coded genetic algorithms. (Standard polynomial mutation operator)
%
% Inputs:
%   P     : D × N matrix            – current population (each column is one individual)
%   lb    : D × 1 vector or scalar  – lower bounds for each dimension
%   ub    : D × 1 vector or scalar  – upper bounds for each dimension
%   pm    : scalar or 1 × N vector  – mutation probability per gene
%                                    scalar  : same probability for all individuals
%                                    1 × N   : individual-specific probability
%                                    default : 1/D (classic Deb choice)
%   scale : scalar                  – distribution index η_m (mutation strength parameter)
%                                    Higher values → smaller perturbations near parent
%                                    Typical values: 20 (common default in NSGA-II literature)
%
% Output:
%   P     : D × N matrix            – mutated population (in-place modification)

    [D, N] = size(P);

    % BUG FIX 1: nargin == 3 时 pm 和 scale 都缺失，需同时赋默认值
    %            原代码 nargin==3 只赋 pm，scale 未定义会在 nm=scale+1 处报错
    if nargin < 4 || isempty(pm)
        pm = 1 / D;       % Classic default: expected one mutation per individual
    end
    if nargin < 5 || isempty(scale)
        scale = 20;       % Standard default distribution index from Deb (1995/2001)
    end

    % Step 1: Generate mutation mask – each gene mutates independently with probability pm
    % scalar pm: rand(D,N) <= pm broadcasts naturally
    % 1×N pm:    expand to D×N for element-wise comparison
    if isscalar(pm)
        mutate_mask = rand(D, N) < pm;
    elseif isequal(size(pm), [1, N])
        mutate_mask = rand(D, N) < pm(ones(D,1), :);  % 1×N → D×N
    else
        error('pm must be a scalar or a 1×N row vector (got size [%s]).', ...
              num2str(size(pm)));
    end

    % Early return if no genes are selected for mutation
    idx_flat = find(mutate_mask);
    if isempty(idx_flat)
        return;
    end

    % Step 2: Extract bounds corresponding to mutated positions
    % BUG FIX 2: lb/ub 为标量时不能用行索引 lb(rows) 取值（索引越界）
    %            原代码对标量 lb/ub 直接用 rows 下标，当 rows 含 >1 的值时报错
    rows = mod(idx_flat - 1, D) + 1;   % row indices of mutated elements
    xi   = P(idx_flat);                % current values at mutated positions

    % BUG FIX 2: lb/ub 为标量时不能用行索引 lb(rows) 取值（索引越界）
    %            标量直接用即可，后续运算 MATLAB 自动广播；非标量才按行索引
    if isscalar(lb)
        lbi = lb;
        ubi = ub;
    else
        lbi = lb(rows);
        ubi = ub(rows);
    end

    range_i = ubi - lbi;

    % Step 3: Normalize distances to boundaries
    d1 = (xi - lbi) ./ range_i;           % normalized distance to lower bound [0,1]
    d2 = (ubi - xi) ./ range_i;           % normalized distance to upper bound [0,1]

    % Precompute exponents (used in both branches)
    nm     = scale + 1;
    inv_nm = 1 / nm;

    % Step 4: Generate perturbation δq using polynomial distribution
    u  = rand(size(xi));
    dq = zeros(size(xi));

    % Branch A: perturbation towards lower bound (u ≤ 0.5)
    % δ_q = (2u + (1-2u)(1-d1)^{η_m+1})^{1/(η_m+1)} - 1
    mask_l = u <= 0.5;
    if any(mask_l)
        u_l        = u(mask_l);
        val        = 2 * u_l + (1 - 2 * u_l) .* (1 - d1(mask_l)).^nm;
        dq(mask_l) = val.^inv_nm - 1;
    end

    % Branch B: perturbation towards upper bound (u > 0.5)
    % δ_q = 1 - (2(1-u) + 2(u-0.5)(1-d2)^{η_m+1})^{1/(η_m+1)}
    mask_r = ~mask_l;
    if any(mask_r)
        u_r        = u(mask_r);
        val        = 2 * (1 - u_r) + 2 * (u_r - 0.5) .* (1 - d2(mask_r)).^nm;
        dq(mask_r) = 1 - val.^inv_nm;
    end

    % Step 5: Apply perturbation and clamp to bounds (handles floating-point issues)
    P(idx_flat) = max(lbi, min(ubi, xi + dq .* range_i));
end