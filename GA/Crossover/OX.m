function O = OX(P1, P2, CR, n)
[D, N] = size(P1);
if nargin < 4
    n = N;
end

O = P1(:, 1:n);

c1 = randi(D, 1, n);
c2 = randi(D, 1, n);
lo = min(c1, c2);
hi = max(c1, c2);

[~, idx_p1] = sort(rand(N, n), 1);
[~, idx_p2] = sort(rand(N, n), 1);
p1_sel = idx_p1(1, :);
p2_sel = idx_p2(1, :);

do_cross = rand(1, n) <= CR;

for i = 1:n
    p1 = P1(:, p1_sel(i))';

    if ~do_cross(i)
        O(:, i) = p1';
        continue
    end

    p2 = P2(:, p2_sel(i))';
    l  = lo(i);
    h  = hi(i);

    % 区间掩码
    in_seg         = false(1, D);
    in_seg(l:h)    = true;

    % 循环遍历顺序：从 h+1 开始绕一圈，共 D-1 个位置
    cycle          = mod((h:h+D-2), D) + 1;  % (1 × D-1)

    % 待填位置 = cycle 中不在区间内的位置
    fill_pos       = cycle(~in_seg(cycle));

    % p2 按 cycle 顺序中不在区间内的元素
    p2_cycle       = p2(cycle);
    fill_vals      = p2_cycle(~in_seg(p2_cycle));

    % 组装子代
    t              = zeros(1, D);
    t(l:h)         = p1(l:h);
    t(fill_pos)    = fill_vals;

    O(:, i) = t';
end
end
