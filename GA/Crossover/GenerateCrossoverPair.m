function pairs = GenerateCrossoverPair(sz, n)
% GenerateCrossoverPair  Generate n crossover parent index pairs (vectorized version)
%
%   pairs = GenerateCrossoverPair(sz, n)
%
% Inputs:
%   sz      Population size (total number of individuals, must be ≥ 2)
%   n       Number of crossover pairs to generate
%
% Output:
%   pairs   2×n integer matrix
%           Each column contains two different integers between 1 and sz
%           (parent indices for crossover)
%
% Description:
%   Generates n pairs of distinct parent indices using randperm (no replacement
%   within the 2n selections). This is fast and vectorized.
%   Note: The same individual may appear in different pairs, which is usually fine.
%
% Example:
%   pairs = GenerateCrossoverPair(100, 20);   % 20 pairs from a population of 100

    if nargin < 2
        error('Two input arguments are required: population size (sz) and number of pairs (n)');
    end
    
    if sz < 2
        error('Population size (sz) must be at least 2');
    end
    
    if n < 1
        pairs = zeros(2, 0);
        return;
    end
    
    % Draw 2*n unique indices from 1 to sz
    all_idx = randperm(sz, 2*n);
    
    % Reshape into 2 rows × n columns
    pairs = reshape(all_idx, 2, n);
end
