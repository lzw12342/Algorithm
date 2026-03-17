function T = BinomialCrossover(P, V, CR)
% BINOMIALCROSSOVER 标准坐标系下的二项式交叉（修复版）
% 输入：
%   P  : 父代种群 (D x N)
%   V  : 变异向量 (D x N)
%   CR : 交叉概率 [0,1]
% 输出：
%   T  : 试验向量 (D x N)

    [D, N] = size(P);
    
    % 修复：原代码 randi(D,N)<=CR 错误，整数与概率无法比较
    % 应使用 rand(D,N) 生成 [0,1] 均匀分布
    mask = rand(D, N) <= CR;
    
    % 确保每列至少一个基因来自V（标准DE的 j_rand 约束）
    j_rand = randi(D, 1, N);
    mask(sub2ind([D, N], j_rand, 1:N)) = true;
    
    % 向量化交叉
    T = P;
    T(mask) = V(mask);
end
