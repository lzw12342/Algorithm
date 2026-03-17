function O = PMX(P, CR, n)
[D, N] = size(P);
if nargin < 3
    n = N;
end

O = zeros(D, n);

c1 = randi(D, 1, n);
c2 = randi(D, 1, n);
lo = min(c1, c2);
hi = max(c1, c2);

[~, idx]  = sort(rand(N, n), 1);
p1_sel    = idx(1, :);
p2_sel    = idx(2, :);
do_cross  = rand(1, n) <= CR;

for i = 1:n
    p = P(:, p1_sel(i))';

    if ~do_cross(i)
        O(:, i) = p';
        continue
    end

    v = P(:, p2_sel(i))';
    l = lo(i);
    h = hi(i);

    % 逆映射表
    inv_v        = zeros(1, D);
    inv_v(v)     = 1:D;

    % in_seg 同时作为 used 的初始状态
    used         = false(1, D);
    used(v(l:h)) = true;

    % 区间从 v 复制
    t        = p;
    t(l:h)   = v(l:h);

    % 区间外：用 find(~in_seg) 替代构造 [1:l-1, h+1:D]
    in_seg        = false(1, D);
    in_seg(l:h)   = true;
    out_pos       = find(~in_seg);        % 预计算区间外位置

    for k = 1:numel(out_pos)
        j   = out_pos(k);
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
