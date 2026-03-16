function T = Crossover_Eigenvector(P, V, CR, eig_p, C)
% EIGENVECTORCROSSOVER 基于特征向量的二项式交叉（含概率控制）
% 输入：
%   P      : 父代种群 (D x N)
%   V      : 变异向量 (D x N)  
%   CR     : 交叉概率 [0,1]
%   params : 结构体，必须包含：
%            .C : 协方差矩阵 (D x D) [仅在特征交叉时使用]
% 输出：
%   T      : 试验向量 (D x N)

    [D, N] = size(P);
    
    % 以概率 params.P 执行特征坐标系交叉（论文Algorithm 1第13行）
    if eig_p >= 1 || rand() < eig_p
        % ===== 特征坐标系交叉（第14-21行） =====
        [B, ~] = eig(C);      % C = B*D^2*B'
        B = real(B);                 % 消除数值误差
        
        % 旋转到特征坐标系（第16行）
        X_prime = B' * P;
        V_prime = B' * V;
        
        % 二项式交叉（第18行）
        mask = rand(D, N) <= CR;
        j_rand = randi(D, 1, N);     % 强制交叉位置
        mask(sub2ind([D, N], j_rand, 1:N)) = true;
        
        U_prime = X_prime;
        U_prime(mask) = V_prime(mask);
        
        % 旋转回原空间（第20行）
        T = B * U_prime;
    else
        % ===== 标准坐标系二项式交叉 =====
        T = P;                       % 默认继承父代
        mask = rand(D, N) <= CR;     % 二项分布掩码
        
        % 强制每列至少一个基因来自V
        j_rand = randi(D, 1, N);
        mask(sub2ind([D, N], j_rand, 1:N)) = true;
        
        T(mask) = V(mask);           % 应用交叉
    end
end