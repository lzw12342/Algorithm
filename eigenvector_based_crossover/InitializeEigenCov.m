function params = InitializeEigenCov(P)
    [D, N] = size(P);
    params.C  = eye(D);
    params.m  = mean(P, 2);      % 论文第8行：初始算术均值
    params.p  = zeros(D, 1);

    % 论文第3-7行：log权重，全部为正
    w_prime      = log(N + 0.5) - log(1:N);   % 全正，i=N时最小但>0
    params.w     = w_prime / sum(w_prime);     % 归一化，和为1
    params.mu_eff = sum(w_prime)^2 / sum(w_prime.^2);

    % 论文第9行
    params.cc = (4 + params.mu_eff/D) / (D + 4 + 2*params.mu_eff/D);
    params.c1 = 2 / ((D + 1.3)^2 + params.mu_eff);
end
