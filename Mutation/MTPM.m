function P = MTPM(P, lb, ub, scale)
    % MTPM - Makinen-Periaux-Toivanen Mutation operator
    %   Applies a non-uniform, boundary-respecting mutation to real-coded variables.
    %   Originally proposed in Mäkinen et al. (1999) for numerical optimization.
    %
    % Inputs:
    %   P       : D × N matrix – current population (each column is one individual)
    %   lb      : D × 1 vector – lower bounds for each dimension
    %   ub      : D × 1 vector – upper bounds for each dimension
    %   scale   : scalar – mutation exponent (controls how strongly mutation prefers
    %                     small changes; larger values → smaller typical perturbations)
    %
    % Output:
    %   P       : D × N matrix – mutated population (same size as input)

    [D, N] = size(P);
    
    % Create dimension index matrix (row index repeated across all individuals)
    dim_idx = repmat((1:D)', 1, N);
    
    % Broadcast lower and upper bounds to match the shape of P
    lbi = lb(dim_idx);    % D × N matrix of lower bounds
    ubi = ub(dim_idx);    % D × N matrix of upper bounds
    
    % Normalize decision variables to the interval [0, 1]
    range_i = ubi - lbi;
    t = (P - lbi) ./ range_i;
    
    % Generate uniform random numbers in [0,1] to decide mutation direction
    r = rand(D, N);
    
    % Two-phase mutation (left and right sides of current position)
    
    % Phase 1: Mutation towards the lower bound (when r < t)
    mask_l = r < t;
    if any(mask_l, 'all')
        tl = t(mask_l);
        delta = (tl - r(mask_l)) ./ max(tl, eps);     % avoid division by zero
        t(mask_l) = tl - tl .* (delta .^ scale);
    end
    
    % Phase 2: Mutation towards the upper bound (when r >= t)
    mask_r = ~mask_l;
    if any(mask_r, 'all')
        tr = t(mask_r);
        one_minus_tr = 1 - tr;
        delta = (r(mask_r) - tr) ./ max(one_minus_tr, eps);   % avoid division by zero
        t(mask_r) = tr + one_minus_tr .* (delta .^ scale);
    end
    
    % Denormalize back to original bounds and enforce hard boundaries
    P = lbi + max(0, min(1, t)) .* range_i;
end