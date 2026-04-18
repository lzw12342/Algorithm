function P = ReflectionBoundary(P, lb, ub)
% Reflection Boundary Handling (Vectorized Implementation)
% Supports lb/ub as column vectors (D×1), row vectors (1×D), or scalars
%
% Input:
%   P  - D×N matrix (D decision variables, N individuals/particles)
%   lb - D×1 or 1×D vector (lower bounds for each dimension) or scalar
%   ub - D×1 or 1×D vector (upper bounds for each dimension) or scalar
%
% Output:
%   P  - D×N matrix after boundary reflection processing

    lb = lb(:);
    ub = ub(:);

    % Lower boundary reflection: out-of-bound positions become 2*lb - P
    mask_lb = P < lb;                           % D×N logical matrix
    if any(mask_lb, 'all')
        P = P .* ~mask_lb + (2*lb - P) .* mask_lb;
    end

    % Upper boundary reflection: out-of-bound positions become 2*ub - P
    mask_ub = P > ub;                           % D×N logical matrix
    if any(mask_ub, 'all')
        P = P .* ~mask_ub + (2*ub - P) .* mask_ub;
    end
end
