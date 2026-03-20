function [O1, O2] = UNDX(P1, P2, P3, alpha, beta)
% UNDX - Unimodal Normal Distribution Crossover (Strict Academic Version)
% Ref: Ono, S. & Kobayashi, S. (1997). A Real-Coded Genetic Algorithm for 
% Function Approximation Using Unimodal Normal Distribution Crossover.
%
% Inputs:
%   P1, P2, P3 : D x n matrices (Parents)
%   Alpha      : Std. dev for primary axis (Recommended: 0.5)
%   BetaParam  : Std. dev for orthogonal space (Recommended: 0.35/sqrt(D))

[D, n] = size(P1);

% --- Default Parameters ---
if nargin < 4 || isempty(alpha), alpha = 0.5; end
if nargin < 5 || isempty(beta), beta = 0.35 / sqrt(D); end

% --- Core Geometry ---
Mid = (P1 + P2) * 0.5;
D_vec = P1 - P2;
D_sq = sum(D_vec.^2, 1);
D_norm = sqrt(D_sq);

% Unit vector of the primary axis
SafeMask = D_norm > 1e-12;
e_d = zeros(D, n);
e_d(:, SafeMask) = D_vec(:, SafeMask) ./ D_norm(SafeMask);

% Distance Dp (Perpendicular distance from P3 to P1-P2 line)
V31 = P3 - P1;
ProjLen = sum(V31 .* e_d, 1);
% Using Pythagorean: Dp = sqrt(|V31|^2 - ProjLen^2)
Dp = sqrt(max(0, sum(V31.^2, 1) - ProjLen.^2));

% --- Offspring Generation ---
% According to the original formula, we need 2 offspring
O1 = Mid + GenerateStrictOffset(e_d, D_norm, Dp, alpha, beta, D, n);
O2 = Mid + GenerateStrictOffset(e_d, D_norm, Dp, alpha, beta, D, n);

end

function Offset = GenerateStrictOffset(e_d, D_norm, Dp, Alpha, BetaParam, D, n)
    % 1. Primary axis component: N(0, (Alpha * L)^2)
    % Note: L = D_norm
    Z1 = randn(1, n) .* (Alpha .* D_norm);
    Longitudinal = Z1 .* e_d;
    
    % 2. Orthogonal components:
    % Original Paper: Sum of (eta_i * Dp * e_i) for i = 1 to D-1
    % Statistically, this is equivalent to generating D-dimensional isotropic 
    % Gaussian noise, projecting it to the orthogonal subspace, and scaling.
    
    Z_raw = randn(D, n); 
    Z_proj = sum(Z_raw .* e_d, 1);
    Z_perp = Z_raw - Z_proj .* e_d; % Vector in (D-1) subspace
    
    % Refined Scaling to match BetaParam:
    % To strictly follow the original distribution, the perturbation in 
    % EACH orthogonal direction must have std. dev = BetaParam * Dp.
    % Our Z_perp already follows this distribution in the subspace.
    Transversal = Z_perp .* (BetaParam .* Dp);
    
    Offset = Longitudinal + Transversal;
end