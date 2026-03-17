function P_new = PMX(P, CR, n)
    [D, N] = size(P);
    if nargin < 3, n = N; end
    
    % 预计算交叉点
    c1 = randi(D, 1, n);
    c2 = randi(D, 1, n);
    lo = min(c1, c2);
    hi = max(c1, c2);
    
    % 样本选择
    [~, idx] = sort(rand(N, n), 1);
    p1_sel = idx(1, :);
    p2_sel = idx(2, :);
    do_cross = rand(1, n) <= CR;
    
    % 预分配输出（直接在原矩阵上操作或建立新矩阵）
    P_new = P(:, p1_sel); 
    
    % 预分配辅助空间，避免循环内重复创建
    inv_v = zeros(1, D);
    used = false(1, D);
    
    for i = 1:n
        if ~do_cross(i), continue; end
        
        % 获取父代个体（直接列操作减少转置）
        p1 = P(:, p1_sel(i))';
        p2 = P(:, p2_sel(i))';
        
        l = lo(i); h = hi(i);
        seg_p2 = p2(l:h);
        
        % 快速重置映射表和状态
        inv_v(p2) = 1:D;
        used(:) = false;
        used(seg_p2) = true;
        
        % 初始化子代
        t = p1;
        t(l:h) = seg_p2;
        
        % 仅处理区间外的位置
        % 优化：直接跳过 [l:h] 范围，减少 find 的调用
        for j = [1:l-1, h+1:D]
            val = p1(j);
            % 冲突处理：只有当 p1 的值已经在 p2 的段中时才需要映射
            while used(val)
                val = p1(inv_v(val));
            end
            t(j) = val;
            % 理论上 PMX 区间外填充不需要 update used，
            % 因为 PMX 保证了映射后的唯一性，但保留以防逻辑变体
        end
        P_new(:, i) = t';
    end
end
