function O = PMX(P, CR, n)
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
    
    % 预分配输出，直接按列继承 p1
    O = P(:, p1_sel); 
    
    % 预分配辅助空间（列向量），避免循环内重复创建内存
    inv_v = zeros(D, 1);
    used = false(D, 1);
    
    for i = 1:n
        if ~do_cross(i), continue; end
        
        % --- 消除转置：直接提取列向量 ---
        p1 = P(:, p1_sel(i));
        p2 = P(:, p2_sel(i));
        
        l = lo(i); h = hi(i);
        seg_p2 = p2(l:h);
        
        % 建立反向映射表 (Inverse Mapping)
        % 映射关系：值 -> 在 p2 中的索引
        inv_v(p2) = 1:D;
        
        % 标记 p2 段中已经存在的基因
        used(:) = false;
        used(seg_p2) = true;
        
        % 初始化子代列向量
        t = p1;
        t(l:h) = seg_p2;
        
        % 处理区间外的冲突 (Mapping phase)
        % 直接遍历预定义的非区间索引，避免使用 find()
        for j = [1:l-1, h+1:D]
            val = p1(j);
            % 如果 p1 这里的基因已经在 p2 的段中了，则产生冲突
            while used(val)
                % 通过映射表找到该值在 p2 段中对应位置在 p1 中的值
                val = p1(inv_v(val));
            end
            t(j) = val;
        end
        
        % --- 直接按列赋值 ---
        O(:, i) = t;
    end
end
