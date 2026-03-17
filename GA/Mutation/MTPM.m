function O = MTPM(P, MR, m_exp)
% MPTMMutation  Makinen-Periaux-Toivanen Mutation (MPTM)
% 原论文公式（Mäkinen et al. 1999, Int J Numer Methods Fluids）
% 输入：
%   P     : (D × N) 当前种群段（lev_r 或 whr_c，已归一化到 [0,1]）
%   p_exp : mutation exponent（推荐 4~10，越大越偏好微小扰动）
%   pm    : 每基因突变概率（推荐 0.05~0.1，和你 DE 的 F 风格一致）
% 输出：
%   O     : (D × N) 突变后的种群段

    [D, N] = size(P);
    O = P;                    % 默认不突变
    
    % 为每个基因独立决定是否突变（标准做法）
    mutate_mask = rand(D, N) < MR;
    
    if ~any(mutate_mask(:)), return; end
    
    % 只对需要突变的基因进行计算
    lb = 0; ub = 1;            % 你当前 lev_r / whr_c 都在 [0,1]（修复后）
    % 如果有不同边界，可改成向量：lb = repmat(lb_vec,1,N) 等
    
    t = (P - lb) ./ (ub - lb);   % 归一化到 [0,1]
    rnd = rand(D, N);
    
    % 原论文公式（已精确还原）
    left  = rnd < t;
    right = rnd > t;
    
    t_m = t;   % 默认不变
    
    % 左侧扰动（rnd < t）
    delta_left = (t(left) - rnd(left)) ./ t(left);
    t_m(left) = t(left) - t(left) .* (delta_left .^ m_exp);
    
    % 右侧扰动（rnd > t）
    delta_right = (rnd(right) - t(right)) ./ (1 - t(right));
    t_m(right) = t(right) + (1 - t(right)) .* (delta_right .^ m_exp);
    
    % 恢复到原边界
    O(mutate_mask) = lb + t_m(mutate_mask) .* (ub - lb);
end
