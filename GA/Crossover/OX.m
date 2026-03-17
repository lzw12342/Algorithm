function O = OX(P, CR, n)
    [D, N] = size(P);
    if nargin < 3, n = N; end
    
    % 随机与选择逻辑保持不变
    c1 = randi(D, 1, n);
    c2 = randi(D, 1, n);
    lo = min(c1, c2);
    hi = max(c1, c2);
    
    [~, idx] = sort(rand(N, n), 1);
    p1_sel = idx(1, :);
    p2_sel = idx(2, :);
    do_cross = rand(1, n) <= CR;
    
    % 预分配输出，默认继承 p1 对应列
    O = P(:, p1_sel); 
    
    % 预分配布尔查找表（列向量）
    in_seg = false(D, 1);
    
    for i = 1:n
        if ~do_cross(i), continue; end
        
        % 直接提取列向量，不转置
        p1 = P(:, p1_sel(i));
        p2 = P(:, p2_sel(i));
        l = lo(i); h = hi(i);
        
        % 1. 提取 p1 保留片段
        seg_p1 = p1(l:h);
        
        % 2. 构造 p2 的填充序列 (纵向拼接保持列向量)
        p2_rotated = [p2(h+1:D); p2(1:h)];
        
        % 3. 快速过滤 (利用线性索引)
        in_seg(:) = false;
        in_seg(seg_p1) = true;
        to_fill = p2_rotated(~in_seg(p2_rotated));
        
        % 4. 直接在列向量 t 上操作
        t = zeros(D, 1);
        t(l:h) = seg_p1;
        
        num_suffix = D - h;
        if num_suffix > 0
            t(h+1:D) = to_fill(1:num_suffix);
            t(1:l-1) = to_fill(num_suffix+1:end);
        else
            % 如果 h 正好是 D，则全部填在前面
            t(1:l-1) = to_fill;
        end
        
        % 赋值回 O 的第 i 列，无需转置
        O(:, i) = t;
    end
end
