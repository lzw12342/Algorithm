function [O1, O2] = PMX(P, PI1, PI2)
% PMX Partially Mapped Crossover (Partially Mapped Crossover)
% Input:
%   P    : D×N population matrix (permutation encoding, each column is an individual)
%   PI1  : 1×n or n×1 vector, indices of first parents
%   PI2  : 1×n or n×1 vector, indices of second parents
% Output:
%   O1   : D×n offspring 1 matrix (template from parent 1 + mapped segment from parent 2)
%   O2   : D×n offspring 2 matrix (template from parent 2 + mapped segment from parent 1)

    D = size(P,1);
    
    % Force row vectors
    PI1 = PI1(:)';  
    PI2 = PI2(:)';  
    n = length(PI1);
    
    if length(PI2) ~= n
        error('PI1 and PI2 must have the same length');
    end
    
    O1 = zeros(D, n);
    O2 = zeros(D, n);
    
    % Reusable auxiliary arrays
    mapping = zeros(D, 1);     % mapping table
    used    = false(D, 1);     % used flags
    
    for i = 1:n
        % Random crossover points
        c1 = randi(D);
        c2 = randi(D);
        left  = min(c1, c2);
        right = max(c1, c2);
        
        p1 = P(:, PI1(i));
        p2 = P(:, PI2(i));
        
        % ──────────────── Offspring 1: base = p1, swap segment from p2 ────────────────
        o1 = p1;
        segment = p2(left:right);
        o1(left:right) = segment;
        
        % Build mapping: value in p2 segment → corresponding value in p1 segment
        mapping(p2(left:right)) = p1(left:right);
        
        used(:) = false;
        used(segment) = true;
        
        % Resolve conflicts outside the segment
        for j = [1:left-1, right+1:D]
            val = p1(j);
            while used(val)
                val = mapping(val);
            end
            o1(j) = val;
            used(val) = true;  % mark as used (optional but safer)
        end
        
        O1(:, i) = o1;
        
        % ──────────────── Offspring 2: base = p2, swap segment from p1 ────────────────
        o2 = p2;
        segment = p1(left:right);
        o2(left:right) = segment;
        
        mapping(p1(left:right)) = p2(left:right);
        
        used(:) = false;
        used(segment) = true;
        
        for j = [1:left-1, right+1:D]
            val = p2(j);
            while used(val)
                val = mapping(val);
            end
            o2(j) = val;
            used(val) = true;
        end
        
        O2(:, i) = o2;
    end
end
