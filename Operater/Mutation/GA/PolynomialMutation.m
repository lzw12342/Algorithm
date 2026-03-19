function P = PolynomialMutation(P, lb, ub, MR, m_eta)
    % 输入：P (DxN), lb (Dx1), ub (Dx1), MR (scalar), m_eta (scalar)
    if nargin < 5 || isempty(m_eta), m_eta = 20; end
    
    [D, N] = size(P);
    
    % 1. 快速掩码筛选 (逻辑索引比线性索引在某些版本下更快，取决于稀疏度)
    mutate_mask = rand(D, N) < MR;
    idx_flat = find(mutate_mask); 
    if isempty(idx_flat), return; end
    
    % 2. 映射物理边界：避免构造 DxN 的大矩阵
    % 利用线性索引反推行号，实现向量化边界提取
    rows = mod(idx_flat - 1, D) + 1; 
    
    xi = P(idx_flat);
    lbi = lb(rows);
    ubi = ub(rows);
    range_i = ubi - lbi;
    
    % 3. 计算 delta1, delta2 (到边界的相对距离)
    % 提前计算 nm 以减少循环内的减法次数
    nm = m_eta + 1;
    inv_nm = 1 / nm;
    
    d1 = (xi - lbi) ./ range_i;
    d2 = (ubi - xi) ./ range_i;
    
    % 4. 计算位移量 delta_q
    u = rand(size(xi));
    dq = zeros(size(xi));
    
    % 分支 A: u <= 0.5 (左侧位移)
    mask_l = u <= 0.5;
    if any(mask_l)
        u_l = u(mask_l);
        % 标准公式还原
        val = 2*u_l + (1 - 2*u_l) .* (1 - d1(mask_l)).^nm;
        dq(mask_l) = val.^inv_nm - 1;
    end
    
    % 分支 B: u > 0.5 (右侧位移)
    mask_r = ~mask_l;
    if any(mask_r)
        u_r = u(mask_r);
        % 标准公式还原
        val = 2*(1 - u_r) + 2*(u_r - 0.5) .* (1 - d2(mask_r)).^nm;
        dq(mask_r) = 1 - val.^inv_nm;
    end
    
    % 5. 就地更新并截断（处理极小的浮点数溢出）
    P(idx_flat) = max(lbi, min(ubi, xi + dq .* range_i));
end
