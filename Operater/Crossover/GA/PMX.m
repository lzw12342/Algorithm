function [O1, O2] = PMX(P, I1, I2, CR)
% PMX Partially Mapped Crossover (with crossover probability)
% Input:
%   P    : D×N population matrix (permutation encoding, each column an individual)
%   I1   : 1×n or n×1 vector, indices of first parents
%   I2   : 1×n or n×1 vector, indices of second parents
%   CR   : crossover rate (probability) ∈ [0,1] or >1 (default = 1 if omitted)
%          if CR < 0 → return parents directly (no crossover, no error)
% Output:
%   O1   : D×n offspring 1 matrix
%   O2   : D×n offspring 2 matrix

    if nargin < 4 || isempty(CR)
        CR = 1;
    end
    
    D = size(P,1);
    
    % Force row vectors
    I1 = I1(:)';  
    I2 = I2(:)';  
    n = length(I1);
    
    if length(I2) ~= n
        error('I1 and I2 must have the same length');
    end
    
    % CR < 0 → 直接返回父代拷贝
    if CR < 0
        O1 = P(:, I1);
        O2 = P(:, I2);
        return;
    end
    
    O1 = zeros(D, n);
    O2 = zeros(D, n);
    
    % 复用辅助数组
    mapping = zeros(D, 1);
    used    = false(D, 1);
    
    % 决定哪些对要进行交叉
    if CR >= 1
        do_crossover = true(1, n);
    else
        do_crossover = rand(1, n) < CR;
    end
    
    for i = 1:n
        p1 = P(:, I1(i));
        p2 = P(:, I2(i));
        
        if ~do_crossover(i)
            O1(:, i) = p1;
            O2(:, i) = p2;
            continue;
        end
        
        % ──────────────── 执行 PMX 交叉 ────────────────
        c1 = randi(D);
        c2 = randi(D);
        left  = min(c1, c2);
        right = max(c1, c2);
        
        % 子代 1：以 p1 为模板，插入 p2 的片段
        o1 = p1;
        segment = p2(left:right);
        o1(left:right) = segment;
        
        mapping(p2(left:right)) = p1(left:right);
        used(:) = false;
        used(segment) = true;
        
        for j = [1:left-1, right+1:D]
            val = p1(j);
            while used(val)
                val = mapping(val);
            end
            o1(j) = val;
            used(val) = true;  % 可选，但更安全
        end
        
        % 子代 2：以 p2 为模板，插入 p1 的片段
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
        
        O1(:, i) = o1;
        O2(:, i) = o2;
    end
end
