function [O1, O2] = OX(P, PI1, PI2)
% OX Order Crossover (OX)
% Input:
%   P    : D×N population matrix (permutation encoding, each column is an individual)
%   PI1  : 1×n or n×1 vector, indices of first parents
%   PI2  : 1×n or n×1 vector, indices of second parents
% Output:
%   O1   : D×n offspring 1 (keeps segment from parent 1, fills from parent 2 in order)
%   O2   : D×n offspring 2 (keeps segment from parent 2, fills from parent 1 in order)

    [D, ~] = size(P);
    
    % Force row vectors
    PI1 = PI1(:)';  
    PI2 = PI2(:)';  
    n = length(PI1);
    
    if length(PI2) ~= n
        error('PI1 and PI2 must have the same length');
    end
    
    O1 = zeros(D, n);
    O2 = zeros(D, n);
    
    % Generate random crossover points for each pair
    C1_rnd = randi(D, 1, n);
    C2_rnd = randi(D, 1, n);
    C1 = min(C1_rnd, C2_rnd);
    C2 = max(C1_rnd, C2_rnd);
    
    % Reusable lookup table
    in_seg = false(D, 1);
    
    for i = 1:n
        c1 = C1(i);
        c2 = C2(i);
        len_suffix = D - c2;
        
        p1 = P(:, PI1(i));
        p2 = P(:, PI2(i));
        
        % ──────────────── Offspring 1: segment from p1, rest from p2 in order ────────────────
        seg = p1(c1:c2);
        
        % Rotated p2 (starting after c2)
        p2_rotated = [p2(c2+1:D); p2(1:c2)];
        
        in_seg(:) = false;
        in_seg(seg) = true;
        to_fill = p2_rotated(~in_seg(p2_rotated));
        
        o = zeros(D, 1);
        o(c1:c2) = seg;
        
        if len_suffix > 0
            o(c2+1:D)     = to_fill(1:len_suffix);
            o(1:c1-1)     = to_fill(len_suffix+1:end);
        else
            o(1:c1-1)     = to_fill;
        end
        O1(:, i) = o;
        
        % ──────────────── Offspring 2: segment from p2, rest from p1 in order ────────────────
        seg = p2(c1:c2);
        
        p1_rotated = [p1(c2+1:D); p1(1:c2)];
        
        in_seg(:) = false;
        in_seg(seg) = true;
        to_fill = p1_rotated(~in_seg(p1_rotated));
        
        o = zeros(D, 1);
        o(c1:c2) = seg;
        
        if len_suffix > 0
            o(c2+1:D)     = to_fill(1:len_suffix);
            o(1:c1-1)     = to_fill(len_suffix+1:end);
        else
            o(1:c1-1)     = to_fill;
        end
        O2(:, i) = o;
    end
end
