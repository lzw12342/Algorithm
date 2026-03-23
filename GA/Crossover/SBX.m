function [O1, O2] = SBX(P1, P2, ni)
% SBX - Simulated Binary Crossover for real-valued variables
%
% Algorithm:
%   1. u ~ Uniform(0, 1)
%   2. beta = (2u)^(1/(ni+1))         if u <= 0.5
%           = (1/(2(1-u)))^(1/(ni+1)) if u >  0.5
%   3. O1 = 0.5 * [(1+beta)*P1 + (1-beta)*P2]
%   4. O2 = 0.5 * [(1-beta)*P1 + (1+beta)*P2]  (only if requested)
%
% Inputs:
%   P1 : (D x N) matrix — first  parent population
%   P2 : (D x N) matrix — second parent population
%   ni : distribution index — scalar or (1 x N) row vector (default = 20)
%        Small ni (e.g.  2– 5) => wide spread, strong exploration.
%        Large ni (e.g. 30–50) => tight spread, strong exploitation.
%
% Outputs:
%   O1 : (D x N) first  offspring — always produced
%   O2 : (D x N) second offspring — only produced when nargout > 1
%
% Usage:
%   O1        = SBX(P1, P2);          % single offspring
%   [O1, O2]  = SBX(P1, P2);          % paired offspring
%   [O1, O2]  = SBX(P1, P2, 10);      % custom scalar ni
%   [O1, O2]  = SBX(P1, P2, ni_vec);  % per-individual ni (1 x N)
% --------------------------------------------------------------------------
% 0. Input validation & defaults
% --------------------------------------------------------------------------
if nargin < 3 || isempty(ni)
    ni = 20;
end
if ~isnumeric(ni) || ~isvector(ni) || any(ni <= 0)
    error('SBX: ni must be a positive scalar or (1 x N) row vector.');
end
if ~isequal(size(P1), size(P2))
    error('SBX: P1 and P2 must have identical dimensions (D x N).');
end
[D, N] = size(P1);
if ~isscalar(ni) && numel(ni) ~= N
    error('SBX: ni as a vector must have length N (one value per individual).');
end
% --------------------------------------------------------------------------
% 1. Sample u ~ Uniform(0,1)
% --------------------------------------------------------------------------
u = rand(D, N);
% --------------------------------------------------------------------------
% 2. Spread factor beta via inverse CDF of SBX distribution
%    scalar ni : e is a plain scalar, MATLAB broadcasts across (D x N)
%    vector ni : repmat to (D x N) so logical indexing works correctly
% --------------------------------------------------------------------------
beta = zeros(D, N);
mask = u <= 0.5;
if isscalar(ni)
    e           = 1 / (ni + 1);
    beta( mask) = (2 .* u( mask))               .^ e;
    beta(~mask) = (1 ./ (2 .* (1 - u(~mask)))) .^ e;
else
    e           = repmat(1 ./ (ni + 1), D, 1);  % (D x N)
    beta( mask) = (2 .* u( mask))               .^ e( mask);
    beta(~mask) = (1 ./ (2 .* (1 - u(~mask)))) .^ e(~mask);
end
% --------------------------------------------------------------------------
% 3. Generate offspring
%    O2 is only computed when the caller requests it (nargout > 1).
% --------------------------------------------------------------------------
O1 = 0.5 * ((1 + beta) .* P1 + (1 - beta) .* P2);
if nargout > 1
    O2 = 0.5 * ((1 - beta) .* P1 + (1 + beta) .* P2);
end
end