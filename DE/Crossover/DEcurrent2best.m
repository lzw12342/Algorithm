function V = DEcurrent2best(P, best, F, n)
% DE/current-to-best/1 变异算子
% 公式：v_i = x_i + F*(x_best - x_i) + F*(x_r1 - x_r2)
% 输入：
%   P    : (D × N) 父代种群
%   best : (D × 1) 最优个体列向量
%   F    : 标量或 (1 × n)
%   n    : 生成子代数量（默认 = N）
% 输出：
%   V : (D × n)

N = size(P, 2);
if nargin < 4
    n = N;
end

[~, rnd] = sort(rand(N, n), 1);   % (N × n)

if n == N
    base = P;                      % (D × N)，x_i 就是自身
    r1   = rnd(1,:);
    r2   = rnd(2,:);
else
    base = P(:, rnd(1,:));         % (D × n)，用第1行作为基向量
    r1   = rnd(2,:);               % r1/r2 后移，避免和 base 重合
    r2   = rnd(3,:);
end

best_dir = best - base;            % (D × n)
diff_dir = P(:,r1) - P(:,r2);     % (D × n)

V = base + F .* best_dir + F .* diff_dir;
end
