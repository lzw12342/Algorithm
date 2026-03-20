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
% Resulting matrix 'cands' is (k x total_selections).
% randi allows for sampling with replacement, standard in GA.
cands = randi(N, k, total_selections);

% 2. Retrieve fitness values for all candidates
cand_fit = P_fit(cands);

% 3. Find the index of the minimum fitness (winner) in each tournament (column).
% 'best_in_tournament' is the row index (1 to k) of the winner in each column.
[~, best_in_tournament] = min(cand_fit, [], 1);

% 4. Extract the population indices of the winners using linear indexing.
% We calculate the linear offset for each column to pick the winner from 'cands'.
column_offsets = 0:k:(k * (total_selections - 1));
I_linear = cands(best_in_tournament + column_offsets);

% 5. Reshape the winner indices to the requested (sz1 x sz2) dimensions.
I = reshape(I_linear, sz1, sz2);

end