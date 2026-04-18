function I = TournamentSelection(P_fit, sz1, sz2, k)
% TournamentSelection - Tournament selection operator for minimization problems.
%
% This function performs sz1 * sz2 independent tournaments. In each tournament,
% k individuals are randomly sampled, and the one with the lowest fitness value
% (best) is selected.
%
% Inputs:
%   P_fit : (1 x N) Vector of fitness values (smaller is better).
%   sz1   : Number of rows in the output matrix.
%   sz2   : Number of columns in the output matrix.
%   k     : Tournament size (number of candidates per selection), default = 2.
%
% Output:
%   I     : (sz1 x sz2) Matrix of selected individual indices.

% Get population size
N = numel(P_fit);

% Default values handling
if nargin < 2 || isempty(sz1), sz1 = 1; end
if nargin < 3 || isempty(sz2), sz2 = N; end
if nargin < 4 || isempty(k),   k = 2;   end

% Total number of selections to perform
total_selections = sz1 * sz2;

% 1. Randomly sample k candidates for each of the total_selections tournaments.
%    Resulting matrix 'cands' is (k x total_selections).
%    ceil(rand*N) is used instead of randi(N) for better performance at large scale,
%    as it avoids the internal modulo operation in randi.
cands = ceil(rand(k, total_selections) * N);

% 2. Retrieve fitness values for all candidates.
%    reshape ensures the result is always (k x total_selections),
%    preventing dimension collapse when total_selections = 1
%    (P_fit being a row vector would otherwise cause P_fit(cands) to
%    return a row vector instead of a column vector).
cand_fit = reshape(P_fit(cands), k, total_selections);

% 3. Find the row index of the minimum fitness (winner) in each tournament (column).
%    'best_in_tournament' is a (1 x total_selections) vector of row indices (1 to k).
[~, best_in_tournament] = min(cand_fit, [], 1);

% 4. Convert (row, column) positions to linear indices into 'cands', then reshape.
%    column_offsets shifts each winner's row index to the correct linear position:
%    column j starts at offset (j-1)*k, so the winner's linear index is
%    best_in_tournament(j) + (j-1)*k.
%    Multiplication is used instead of the colon step syntax (0:k:...) for clarity.
column_offsets = (0:total_selections-1) * k;

% 5. Index into 'cands' with linear indices to extract winners, then reshape
%    to the requested (sz1 x sz2) output dimensions.
I = reshape(cands(best_in_tournament + column_offsets), sz1, sz2);
end