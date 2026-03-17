function T = PMX(P, V)
% PMX（Partially Mapped Crossover）排列交叉
% 输入：
%   P : (n × N) 父代排列矩阵，每列一个排列
%   V : (n × N) 变异排列矩阵，每列一个排列
% 输出：
%   T : (n × N) 子代排列矩阵
%
% 优化：
%   预建逆映射表 inv_v，将映射链查询从 O(n) 降到 O(1)
%   used 用逻辑数组替代 set，索引更快
%   按行向量操作，减少转置开销

[n, N] = size(P);
T = P;

% 每个个体独立随机选交叉区间 [lo, hi]
c1 = randi(n, 1, N);
c2 = randi(n, 1, N);
lo = min(c1, c2);                        % (1 × N)
hi = max(c1, c2);                        % (1 × N)

for i = 1:N
    p = P(:, i)';                        % (1 × n) 行向量
    v = V(:, i)';                        % (1 × n) 行向量
    t = p;                               % 先复制父代

    l = lo(i);
    h = hi(i);

    % 预建逆映射表：inv_v(val) = val 在 v 中的位置，O(n) 建表 O(1) 查询
    inv_v = zeros(1, n);
    inv_v(v) = 1:n;

    % used 逻辑数组，值域 1~n
    used = false(1, n);

    % 步骤1：区间直接从 v 复制
    t(l:h)      = v(l:h);
    used(v(l:h)) = true;

    % 步骤2：区间外从 p 继承，冲突时沿映射链替换
    for j = [1:l-1, h+1:n]
        val = p(j);
        while used(val)
            val = p(inv_v(val));         % O(1) 逆映射查询
        end
        t(j)      = val;
        used(val) = true;
    end

    T(:, i) = t';                        % 写回列向量
end
end
