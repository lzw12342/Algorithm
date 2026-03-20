function [O1, O2] = LaplaceX(P1, P2, b, a)
% LaplaceX - Batch Real-coded Laplace Crossover (LX) operator
%
% Inputs:
%   P1 : (D x n) matrix, first parents
%   P2 : (D x n) matrix, second parents
%   b  : Scale parameter (controls spread, default = 0.1)
%   a  : Location parameter (default = 0)
%
% Outputs:
%   O1, O2 : (D x n) offspring matrices

    % Default values handling
    if nargin < 3 || isempty(b), b = 0.1; end
    if nargin < 4 || isempty(a), a = 0;   end

    % Validate dimensions
    if ~isequal(size(P1), size(P2))
        error('P1 and P2 must have the same dimensions (D x n)');
    end
    [D, n] = size(P1);

    % Generate uniform random numbers for two independent sets of beta
    % We generate two sets so O1 and O2 explore different points in space
    u = rand(D, n);
    r = rand(D, n);
    
    % --- Calculate beta for Offspring 1 ---
    beta1 = zeros(D, n);
    idx_low  = r <= 0.5;
    idx_high = ~idx_low;
    
    % Laplace Distribution Inverse CDF transformation
    beta1(idx_low)  = a - b * log(u(idx_low));
    beta1(idx_high) = a + b * log(u(idx_high));

    % --- Calculate beta for Offspring 2 (Optional: use fresh randoms) ---
    % Standard LX often uses different random numbers for the second child
    u2 = rand(D, n);
    r2 = rand(D, n);
    beta2 = zeros(D, n);
    idx_low2  = r2 <= 0.5;
    idx_high2 = ~idx_low2;
    
    beta2(idx_low2)  = a - b * log(u2(idx_low2));
    beta2(idx_high2) = a + b * log(u2(idx_high2));

    % Common distance scale
    dist = abs(P1 - P2);
    
    % Generate offspring
    O1 = P1 + beta1 .* dist;
    O2 = P2 + beta2 .* dist;
end