function [O1, O2] = LogisticX(P1, P2, s)
% LogisticX - Logistic Distribution-based Crossover Operator (LogX)
% Strictly follows Naqvi et al. (2020), Journal of Statistical Computation
% and Simulation. DOI: 10.1080/00949655.2020.1832093
%
% Algorithm (Equations 17-19 in the paper):
%   1. u ~ Uniform(0, 1)
%   2. beta = s * log(u/(1-u))   [Inverse CDF, mu=0]
%   3. O1 = 0.5 * [(P1+P2) + beta .* |P1-P2|]   (Eq. 18)
%   4. O2 = 0.5 * [(P1+P2) - beta .* |P1-P2|]   (Eq. 19, only if requested)
%
% Inputs:
%   P1  : (D x N) matrix — first  parent population
%   P2  : (D x N) matrix — second parent population
%   s   : scalar scale parameter of Logistic distribution (default = 1.0)
%         Typical range: 0.5 – 5.0.
%         Small s (e.g. 0.5) => offspring cluster near parental midpoint (exploitation).
%         Large s (e.g. 5.0) => offspring can deviate far from parents  (exploration).
%         Values outside [0.1, 10] are rarely useful in practice.
%
% Outputs:
%   O1  : (D x N) first  offspring — always produced
%   O2  : (D x N) second offspring — only produced when nargout > 1
%
% Usage:
%   O1        = LogisticX(P1, P2);       % single offspring, saves compute
%   [O1, O2]  = LogisticX(P1, P2);       % paired offspring (default paper mode)
%   [O1, O2]  = LogisticX(P1, P2, 2.5);  % custom scale

% --------------------------------------------------------------------------
% 0. Input validation & defaults
% --------------------------------------------------------------------------
if nargin < 3 || isempty(s)
    s = 1.0;
end

if ~isnumeric(s) || ~isvector(s) || any(s <= 0)
    error('LogisticX: scale parameter s must be a positive scalar or (1 x N) row vector.');
end

if ~isequal(size(P1), size(P2))
    error('LogisticX: P1 and P2 must have identical dimensions (D x N).');
end

% --------------------------------------------------------------------------
% 1. Pre-compute midpoint and absolute distance
% --------------------------------------------------------------------------
mid  = 0.5 * (P1 + P2);
dist = abs(P1 - P2);

% --------------------------------------------------------------------------
% 2. Sample u ~ Uniform(0,1) with double-sided clip for numerical safety
% --------------------------------------------------------------------------
u = rand(size(P1));
u = max(1e-16, min(1 - 1e-16, u));

% --------------------------------------------------------------------------
% 3. Inverse-CDF transform (Eq. 17, mu = 0)
%    beta = s * log(u / (1-u))
%    s can be scalar or (1 x N) row vector
% --------------------------------------------------------------------------
if isscalar(s)
    beta = s * log(u ./ (1 - u));
else
    % s is (1 x N): broadcast across D rows
    beta = s .* log(u ./ (1 - u));   % (D x N) .* (1 x N) → (D x N)
end

% --------------------------------------------------------------------------
% 4. Compute half-step (shared by both offspring)
% --------------------------------------------------------------------------
half_step = 0.5 * beta .* dist;

% --------------------------------------------------------------------------
% 5. Generate offspring
%    O2 is only computed when the caller requests it (nargout > 1).
%    When only O1 is needed, the subtraction and memory allocation for O2
%    are skipped entirely — no wasted computation.
% --------------------------------------------------------------------------
O1 = mid + half_step;

if nargout > 1
    O2 = mid - half_step;
end

end