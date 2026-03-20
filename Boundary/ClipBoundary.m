function P = ClipBoundary(P, lb, ub)
% Clip - Vectorized Boundary Clipping (Saturation)
% Supports lb/ub as vectors (D×1 or 1×D) or scalars.
%
% Input:
%   P  - D×N matrix (D decision variables, N individuals)
%   lb - Lower bounds (scalar or vector)
%   ub - Upper bounds (scalar or vector)
%
% Output:
%   P  - D×N matrix where out-of-bounds values are clamped to [lb, ub]

    % Ensure lb and ub are column vectors (D×1) for automatic broadcasting
    lb = lb(:);
    ub = ub(:);

    % Use MATLAB's built-in min/max for the fastest vectorized clipping
    % max(P, lb) sets all values below lb to lb
    % min(..., ub) sets all values above ub to ub
    P = max(min(P, ub), lb);
    
end