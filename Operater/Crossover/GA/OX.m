function [O1, O2] = OX(P, I1, I2, CR)
% OX Order Crossover (OX) with crossover probability
% Input:
%   P    : D×N population matrix (permutation encoding, each column is an individual)
%   I1   : 1×n or n×1 vector, indices of first parents
%   I2   : 1×n or n×1 vector, indices of second parents
%   CR   : crossover rate (probability) ∈ [0,1] or >1 (default = 1 if omitted)
%          if CR < 0 → return parents directly (no crossover, no error)
% Output:
%   O1   : D×n offspring 1 (segment from parent 1 + ordered fill from parent 2)
%   O2   : D×n offspring 2 (segment from parent 2 + ordered fill from parent 1)

    if nargin < 4 || isempty(CR)
        CR = 1;
    end
    
    D = size(P,1);
    I1 = I1(:)';  
    I2 = I2(:)';  
    n  = length(I1);
    
    if length(I2) ~= n
        error('I1 and I2 must have the same length');
    end
    
    % 特殊情况：CR < 0 → 直接返回父代拷贝，不做任何操作
    if CR < 0
        O1 = P(:, I1);
        O2 = P(:, I2);
        return;
    end
    
    O1 = zeros(D, n);
    O2 = zeros(D, n);
    
    % 预先生成所有可能的交叉点（即使 CR>=1 也会生成，但开销很小）
    C1_rnd = randi(D, 1, n);
    C2_rnd = randi(D, 1, n);
    C1 = min(C1_rnd, C2_rnd);
    C2 = max(C1_rnd, C2_rnd);
    
    % 复用查找表
    in_seg = false(D, 1);
    
    % 决定哪些对要进行交叉
    if CR >= 1
        do_crossover = true(1, n);
    else
        do_crossover = rand(1, n) < CR;
    end
    
    for i = 1:n
        if ~do_crossover(i)
            O1(:, i) = P(:, I1(i));
            O2(:, i) = P(:, I2(i));
            continue;
        end
        
        % ──────────────── 执行 OX 交叉 ────────────────
        p1 = P(:, I1(i));
        p2 = P(:, I2(i));
        
        c1 = C1(i);
        c2 = C2(i);
        len_suffix = D - c2;
        
        % 子代 1
        seg = p1(c1:c2);
        p2_rotated = [p2(c2+1:D); p2(1:c2)];
        in_seg(:) = false;
        in_seg(seg) = true;
        to_fill = p2_rotated(~in_seg(p2_rotated));
        
        o = zeros(D, 1);
        o(c1:c2) = seg;
        if len_suffix > 0
            o(c2+1:D)   = to_fill(1:len_suffix);
            o(1:c1-1)   = to_fill(len_suffix+1:end);
        else
            o(1:c1-1)   = to_fill;
        end
        O1(:, i) = o;
        
        % 子代 2
        seg = p2(c1:c2);
        p1_rotated = [p1(c2+1:D); p1(1:c2)];
        in_seg(:) = false;
        in_seg(seg) = true;
        to_fill = p1_rotated(~in_seg(p1_rotated));
        
        o = zeros(D, 1);
        o(c1:c2) = seg;
        if len_suffix > 0
            o(c2+1:D)   = to_fill(1:len_suffix);
            o(1:c1-1)   = to_fill(len_suffix+1:end);
        else
            o(1:c1-1)   = to_fill;
        end
        O2(:, i) = o;
    end
end
