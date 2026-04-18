function [O1, O2] = PMX(P1, P2)
% PMX - Partially Mapped Crossover for permutation-based encoding
%
%   [O1, O2] = PMX(P1, P2)
%
% Inputs:
%   P1  : (D x n) parent 1 matrix, each column is a permutation individual
%   P2  : (D x n) parent 2 matrix, each column is a permutation individual
%
% Outputs:
%   O1  : (D x n) offspring 1 matrix
%   O2  : (D x n) offspring 2 matrix
%
% Algorithm:
%   1. Randomly select two cut points [left, right] for each pair
%   2. Swap the segments between parents to form initial offspring
%   3. Resolve conflicts using mapping relationships:
%      - For positions outside the segment, if value conflicts with segment,
%        replace it using the mapping chain until no conflict exists

    [D, n] = size(P1);
    
    if ~isequal(size(P1), size(P2))
        error('P1 and P2 must have the same dimensions (D x n)');
    end
    
    O1 = zeros(D, n);
    O2 = zeros(D, n);
    
    % Reusable auxiliary arrays for conflict resolution
    mapping = zeros(D, 1);
    used = false(D, 1);
    
    for i = 1:n
        p1 = P1(:, i);
        p2 = P2(:, i);
        
        % Randomly select two crossover points
        c1 = randi(D);
        c2 = randi(D);
        left = min(c1, c2);
        right = max(c1, c2);
        
        % --- Offspring 1: template p1, insert segment from p2 ---
        o1 = p1;
        segment = p2(left:right);
        o1(left:right) = segment;
        
        % Build mapping: p2 segment values -> p1 segment values
        mapping(p2(left:right)) = p1(left:right);
        
        % Mark segment values as used (occupied)
        used(:) = false;
        used(segment) = true;
        
        % Resolve conflicts outside the segment
        for j = [1:left-1, right+1:D]
            val = p1(j);
            % Follow mapping chain until finding unused value
            while used(val)
                val = mapping(val);
            end
            o1(j) = val;
            used(val) = true;
        end
        
        % --- Offspring 2: template p2, insert segment from p1 ---
        o2 = p2;
        segment = p1(left:right);
        o2(left:right) = segment;
        
        % Build reverse mapping: p1 segment values -> p2 segment values
        mapping(p1(left:right)) = p2(left:right);
        
        % Mark segment values as used
        used(:) = false;
        used(segment) = true;
        
        % Resolve conflicts outside the segment
        for j = [1:left-1, right+1:D]
            val = p2(j);
            while used(val)
                val = mapping(val);
            end
            o2(j) = val;
            used(val) = true;
        end
        
        O1(:, i) = o1;
        O2(:, i) = o2;
    end
end