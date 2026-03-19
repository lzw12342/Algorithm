function [O1, O2] = SBX(P1, P2, ni)
% SBX - Simulated Binary Crossover for real-valued variables
%
% Inputs:
%   P1 : D × n matrix, first parents
%   P2 : D × n matrix, second parents
%   ni : distribution index (controls offspring spread around parents)
%        recommended values:
%          2  – 5   wide spread, strong exploration
%          10 – 20  moderate (default 20, most common in NSGA-II)
%          30 – 50  tight spread, strong exploitation / fine-tuning
%
% Outputs:
%   O1, O2 : two offspring matrices (D × n)

if nargin < 3 || isempty(ni)
    ni = 20;
end

% Validate dimensions
if ~isequal(size(P1), size(P2))
    error('P1 and P2 must have the same dimensions (D × n)');
end

[D, n] = size(P1);

% Uniform random numbers for beta computation
u = rand(D, n);

% Compute spread factor beta via inverse CDF of SBX distribution
beta          = zeros(D, n);
mask          = u <= 0.5;
beta( mask)   = (2 .* u( mask))                   .^ (1 / (ni + 1));
beta(~mask)   = (1 ./ (2 .* (1 - u(~mask))))      .^ (1 / (ni + 1));

% Generate two symmetric offspring around the parents
O1 = 0.5 * ((1 + beta) .* P1 + (1 - beta) .* P2);
O2 = 0.5 * ((1 - beta) .* P1 + (1 + beta) .* P2);

end