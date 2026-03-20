function [O1, O2] = UNDX(P1, P2, P3, sigma_xi, sigma_eta)
% UNDX - Unimodal Normal Distribution Crossover
% Strictly follows Ono, Kita & Kobayashi (2003), "A Real-coded Genetic
% Algorithm using the Unimodal Normal Distribution Crossover", Advances
% in Evolutionary Computing, Springer-Verlag. (Eq. 1-4, Appendix A)
%
% Algorithm (Appendix A):
%   1. t = randn(D,N) .* (D_perp * sigma_eta)        [isotropic draw]
%   2. t = t - (t · e0) * e0                          [project to perp subspace]
%   3. t = t + xi * d,  xi ~ N(0, sigma_xi^2)         [add parallel component]
%   4. O1 = x^p + t  [,  O2 = x^p - t  if nargout > 1]
%
% Inputs:
%   P1, P2 : (D x N) matrix — primary parent pair
%   P3     : (D x N) matrix — third parent, randomly drawn from population
%            Used only to compute D_perp (Eq. 2); never interpolated.
%   sigma_xi  : std. dev for parallel component along d = P2 - P1  (default = 0.5)
%               Eq. 4: sigma_xi^2 = 1/4.  At this value the offspring covariance
%               equals the parent covariance (Theorem 2, Eq. 7).
%               Typical range: [0.1, 2.0].
%               < 0.5 — offspring cluster near x^p along d; covariance shrinks.
%               = 0.5 — covariance-preserving optimum (recommended).
%               > 0.5 — offspring spread further along d; covariance inflates.
%   sigma_eta : std. dev for orthogonal components per unit D_perp  (default = 0.35/sqrt(D))
%               Eq. 4: sigma_eta^2 = 0.35^2/D.  The 1/sqrt(D) factor removes
%               dimension-dependence of orthogonal spread (Eq. 16, Sec. 5.5).
%               Typical range: [0.05/sqrt(D), 2/sqrt(D)].
%               Small  — offspring confined near primary search line.
%               = 0.35/sqrt(D) — empirically optimal (Sec. 7).
%               Large  — orthogonal spread dominates; epistatic performance degrades.
%
% Outputs:
%   O1 : (D x N) — first  offspring = x^p + t        (always produced)
%   O2 : (D x N) — second offspring = x^p - t        (only if nargout > 1)
%
% Usage:
%   O1       = UNDX(P1, P2, P3);              % single offspring
%   [O1, O2] = UNDX(P1, P2, P3);              % paired offspring (paper default)
%   [O1, O2] = UNDX(P1, P2, P3, 0.5, 0.35/sqrt(size(P1,1)));

% --------------------------------------------------------------------------
% 0. Defaults
% --------------------------------------------------------------------------
[D, N] = size(P1);
if nargin < 4 || isempty(sigma_xi),  sigma_xi  = 0.5;            end
if nargin < 5 || isempty(sigma_eta), sigma_eta = 0.35 / sqrt(D); end

% --------------------------------------------------------------------------
% 1. Midpoint and primary search direction  (Eq. 1)
% --------------------------------------------------------------------------
x_p = 0.5 * (P1 + P2);
d   = P2 - P1;

% --------------------------------------------------------------------------
% 2. Unit vector e0 along d; guarded against identical parents
% --------------------------------------------------------------------------
d_norm          = sqrt(sum(d .^ 2, 1));
e0              = zeros(D, N);
safe            = d_norm > 1e-12;
e0(:, safe)     = d(:, safe) ./ d_norm(safe);

% --------------------------------------------------------------------------
% 3. Perpendicular distance D_perp from P3 to the P1-P2 line  (Eq. 2)
% --------------------------------------------------------------------------
V      = P3 - P1;
D_perp = sqrt(max(0, sum(V .^ 2, 1) - sum(V .* e0, 1) .^ 2));

% --------------------------------------------------------------------------
% 4. Isotropic draw then project to (D-1)-dim subspace perpendicular to d
%    (Appendix A, steps 1-2)
%    Equivalent to summing eta_i * D_perp * e_i over i=1..D-1 (Eq. 3),
%    without explicitly constructing the ONB.
% --------------------------------------------------------------------------
t = randn(D, N) .* (sigma_eta .* D_perp);
t = t - sum(t .* e0, 1) .* e0;

% --------------------------------------------------------------------------
% 5. Add parallel component  (Appendix A, step 3)
% --------------------------------------------------------------------------
t = t + randn(1, N) .* sigma_xi .* d;

% --------------------------------------------------------------------------
% 6. Offspring  (Appendix A, step 4)
%    O2 is only allocated when the caller requests it.
% --------------------------------------------------------------------------
O1 = x_p + t;
if nargout > 1
    O2 = x_p - t;
end
end