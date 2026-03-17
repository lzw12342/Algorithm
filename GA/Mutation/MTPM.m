function P = MTPM(P, lb, ub, MR, m_exp)
    [D, N] = size(P);
    
    % 1. 预先确定变异掩码（性能第一步：只处理需要变异的元素）
    mutate_mask = rand(D, N) < MR;
    if ~any(mutate_mask, 'all'), return; end
    
    % 2. 向量化索引优化：避免在大矩阵上使用 repmat
    % 获取需要变异的元素在 P 中的线性索引
    idx_flat = find(mutate_mask);
    [rows, ~] = ind2sub([D, N], idx_flat); % 获取对应的维度索引
    
    % 提取设计变量及其物理边界
    xi_phys = P(idx_flat);
    lbi = lb(rows);
    ubi = ub(rows);
    
    % 3. 归一化（就地计算，减少中间变量）
    range_i = ubi - lbi;
    t = (xi_phys - lbi) ./ range_i;
    
    % 4. 核心算子优化
    r = rand(size(t));
    t_new = t;
    
    % 情况 A: r < t (向左变异)
    mask_l = r < t;
    if any(mask_l)
        tl = t(mask_l);
        % 原公式还原：t_new = t - t * ((t - r) / t)^m_exp
        % 性能优化：利用点幂运算
        t_new(mask_l) = tl - tl .* ((tl - r(mask_l)) ./ max(tl, eps)) .^ m_exp;
    end
    
    % 情况 B: r >= t (向右变异)
    mask_r = ~mask_l;
    if any(mask_r)
        tr = t(mask_r);
        % 原公式还原：t_new = t + (1 - t) * ((r - t) / (1 - t))^m_exp
        one_minus_tr = 1 - tr;
        t_new(mask_r) = tr + one_minus_tr .* ((r(mask_r) - tr) ./ max(one_minus_tr, eps)) .^ m_exp;
    end
    
    % 5. 反归一化并写回（直接在索引位置修改，避免全矩阵操作）
    P(idx_flat) = lbi + max(0, min(1, t_new)) .* range_i;
end
