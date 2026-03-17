function O = PMX(P1, P2, CR, n)
% PMX（Partially Mapped Crossover）排列交叉
% 输入：
%   P  : (D × N) 父代排列矩阵
%   O  : (D × N) 另一亲本排列矩阵
%   CR : 交叉概率 [0,1]，每个个体以概率 CR 执行 PMX，否则直接继承父代
%   n  : 返回子代数量（默认 = N）
% 输出：
%   P_new : (D × n) 子代排列矩阵

[D, N] = size(P1);
if nargin < 4
    n = N;
end

O = P1(:, 1:n);

% 随机选交叉区间
c1 = randi(D, 1, n);
c2 = randi(D, 1, n);
lo = min(c1, c2);
hi = max(c1, c2);

% 随机选亲本列索引
[~, idx_p] = sort(rand(N, n), 1);
[~, idx_o] = sort(rand(N, n), 1);
p_sel = idx_p(1, :);
o_sel = idx_o(1, :);

% CR 掩码：决定哪些个体执行 PMX，哪些直接继承
do_cross = rand(1, n) <= CR;

for i = 1:n
    p = P1(:, p_sel(i))';                % (1 × D)

    if ~do_cross(i)
        O(:, i) = p';
        continue
    end

    v = P2(:, o_sel(i))';                % (1 × D)
    t = p;
    l = lo(i);
    h = hi(i);

    inv_v        = zeros(1, D);
    inv_v(v)     = 1:D;
    used         = false(1, D);
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

    O(:, i) = t';
end
end
