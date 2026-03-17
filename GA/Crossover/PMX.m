function P_new = PMX(P, O, n)
% PMX（Partially Mapped Crossover）排列交叉
% 输入：
%   P       : (n × N) 父代排列矩阵，每列一个排列
%   V       : (n × N) 变异排列矩阵，每列一个排列
%   n_cross : 执行交叉的个体数量（默认 = N）
% 输出：
%   T : (n × n_cross) 子代排列矩阵

[D, N] = size(P);
if nargin < 3
    n = N;
end

P_new = P(:, 1:n);

% 随机选交叉区间
c1 = randi(D, 1, n);
c2 = randi(D, 1, n);
lo = min(c1, c2);
hi = max(c1, c2);

% 随机从 P 和 V 各选 n_cross 列作为两个亲本
[~, idx_p] = sort(rand(N, n), 1);
[~, idx_v] = sort(rand(N, n), 1);
p_sel = idx_p(1, :);                     % (1 × n_cross) 从P选的列索引
v_sel = idx_v(1, :);                     % (1 × n_cross) 从V选的列索引

for i = 1:n
    p = P(:, p_sel(i))';                 % (1 × n)
    v = O(:, v_sel(i))';                 % (1 × n)
    t = p;

    l = lo(i);
    h = hi(i);

    inv_v = zeros(1, D);
    inv_v(v) = 1:D;

    used = false(1, D);

    t(l:h)       = v(l:h);
    used(v(l:h)) = true;

    for j = [1:l-1, h+1:D]
        val = p(j);
        while used(val)
            val = p(inv_v(val));
        end
        t(j)      = val;
        used(val) = true;
    end

    P_new(:, i) = t';
end
end
