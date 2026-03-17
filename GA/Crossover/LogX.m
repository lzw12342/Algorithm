function O = LogX(P, CR, n, eta)
% LOGXCROSSOVER  Logistic Crossover (LogX) - 严格按 Naqvi 2021 论文公式
% 参数：
%   P   : (D × N) 当前种群段
%   eta : scale 参数 s（推荐默认 1.0，论文常用值）
%   CR  : 交叉概率 [0,1]（推荐 0.9，和你 DE 保持一致）
%   n   : 生成子代数量（默认 = N）
% 输出：
%   O   : (D × n) 子代

    [D, N] = size(P);
    if nargin < 3 || isempty(n),   n = N; end
    if nargin < 4 || isempty(eta), eta = 1.0; end

    % ==================== 为每个子代决定是否交叉 ====================
    mask = rand(1, n) <= CR;          % 1×n 的布尔掩码
    num_cross = nnz(mask);
    
    O = zeros(D, n);
    
    % ---------- 不交叉的部分：直接复制随机父代 ----------
    if any(~mask)
        idx_copy = randi(N, 1, nnz(~mask));
        O(:, ~mask) = P(:, idx_copy);
    end
    
    % ---------- 交叉的部分：严格按论文公式 ----------
    if num_cross > 0
        [~, rnd] = sort(rand(N, num_cross + 2), 1);
        p1 = rnd(1, 1:num_cross);
        p2 = rnd(2, 1:num_cross);
        same = (p1 == p2);
        if any(same), p2(same) = rnd(3, same); end
        
        P1 = P(:, p1);
        P2 = P(:, p2);
        
        u = rand(D, num_cross);                    % 每维独立随机数
        
        % 论文精确公式：β = μ - s * log((1-u)/u)，μ=0
        beta = - eta * log((1 - u) ./ u);         % β ∈ (-∞, +∞)
        
        avg  = 0.5 * (P1 + P2);
        delta = abs(P1 - P2);                     % 论文用 |x-y|（不是 0.5倍）
        
        % 生成两个子代中的一个（论文生成 η 和 ξ，这里返回 η 风格）
        O_cross = avg + beta .* delta;            % 你也可以随机选 + 或 -
        
        O(:, mask) = O_cross;
    end
end
