function [O1, O2] = OX(P1, P2)
% OX - Order Crossover operator for permutation-based encoding
%
%   [O1, O2] = OX(P1, P2)
%
% Inputs:
%   P1  : (D x n) parent 1 matrix, each column is a permutation individual
%   P2  : (D x n) parent 2 matrix, each column is a permutation individual
%
% Outputs:
%   O1  : (D x n) offspring 1, inherits segment from P1, fills rest from P2
%   O2  : (D x n) offspring 2, inherits segment from P2, fills rest from P1
%
% Algorithm:
%   1. Randomly select two cut points c1, c2 for each pair
%   2. Offspring 1: copy segment [c1:c2] from P1, fill remaining positions 
%      with elements from P2 (in order, excluding already copied ones)
%   3. Offspring 2: copy segment [c1:c2] from P2, fill remaining positions 
%      with elements from P1 (in order, excluding already copied ones)

    [D, n] = size(P1);
    
    if ~isequal(size(P1), size(P2))
        error('P1 and P2 must have the same dimensions (D x n)');
    end
    
    O1 = zeros(D, n);
    O2 = zeros(D, n);
    
    % Generate random crossover points for all pairs
    C1_rnd = randi(D, 1, n);
    C2_rnd = randi(D, 1, n);
    C1 = min(C1_rnd, C2_rnd);
    C2 = max(C1_rnd, C2_rnd);
    
    % Reusable lookup buffer
    in_seg = false(D, 1);
    
    for i = 1:n
        p1 = P1(:, i);
        p2 = P2(:, i);
        
        c1 = C1(i);
        c2 = C2(i);
        len_suffix = D - c2;
        
        % --- Offspring 1: segment from p1, fill from p2 ---
        seg = p1(c1:c2);
        % Rotate p2 to start from c2+1
        p2_rotated = [p2(c2+1:D); p2(1:c2)];
        
        in_seg(:) = false;
        in_seg(seg) = true;
        to_fill = p2_rotated(~in_seg(p2_rotated));
        
        o = zeros(D, 1);
        o(c1:c2) = seg;
        if len_suffix > 0
            o(c2+1:D) = to_fill(1:len_suffix);
            o(1:c1-1) = to_fill(len_suffix+1:end);
        else
            o(1:c1-1) = to_fill;
        end
        O1(:, i) = o;
        
        % --- Offspring 2: segment from p2, fill from p1 ---
        seg = p2(c1:c2);
        % Rotate p1 to start from c2+1
        p1_rotated = [p1(c2+1:D); p1(1:c2)];
        
        in_seg(:) = false;
        in_seg(seg) = true;
        to_fill = p1_rotated(~in_seg(p1_rotated));
        
        o = zeros(D, 1);
        o(c1:c2) = seg;
        if len_suffix > 0
            o(c2+1:D) = to_fill(1:len_suffix);
            o(1:c1-1) = to_fill(len_suffix+1:end);
        else
            o(1:c1-1) = to_fill;
        end
        O2(:, i) = o;
    end
end