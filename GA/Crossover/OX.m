function O = OX(P, CR, n)
[D, N] = size(P);
if nargin < 3
    n = N;
end

O = zeros(D, n);

c1 = randi(D, 1, n);
c2 = randi(D, 1, n);
lo = min(c1, c2);
hi = max(c1, c2);

[~, idx] = sort(rand(N, n), 1);
p1_sel   = idx(1, :);
p2_sel   = idx(2, :);
do_cross = rand(1, n) <= CR;

% 预分配复用的临时数组，避免循环内重复申请内存
in_seg = false(1, D);
cycle  = zeros(1, D-1);

for i = 1:n
    p1 = P(:, p1_sel(i))';

    if ~do_cross(i)
        O(:, i) = p1';
        continue
    end

    p2 = P(:, p2_sel(i))';
    l  = lo(i);
    h  = hi(i);

    % 复用预分配数组，reset 只改动的部分
    in_seg(:)   = false;
    in_seg(l:h) = true;

    cycle(:)    = mod((h:h+D-2), D) + 1;

    % out_mask 只算一次，供 fill_pos 和 fill_vals 共用
    out_mask    = ~in_seg(cycle);        % (1 × D-1) logical

    t           = zeros(1, D);
    t(l:h)      = p1(l:h);
    t(cycle(out_mask)) = p2(cycle(out_mask));

    O(:, i) = t';
end
end
