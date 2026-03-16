function params = UpdateEigenCov(P, P_Fit, params)
    [D, N] = size(P);

    % 权重长度自适应
    if length(params.w) ~= N
        w_prime      = log(N + 0.5) - log(1:N);
        params.w     = w_prime / sum(w_prime);
        params.mu_eff = sum(w_prime)^2 / sum(w_prime.^2);
        params.cc    = (4 + params.mu_eff/D) / (D + 4 + 2*params.mu_eff/D);
        params.c1    = 2 / ((D + 1.3)^2 + params.mu_eff);
    end

    % 论文第26行：按fitness排序后加权均值
    [~, rank_idx] = sort(P_Fit, 'ascend');   % 最小化，越小排名越靠前
    P_sorted = P(:, rank_idx);
    m_new    = P_sorted * params.w';          % 加权均值，rank1权重最大

    % 论文公式3.5：σ_g = 1，不归一化diff
    diff  = m_new - params.m;
    p_new = (1 - params.cc) * params.p + ...
            sqrt(params.cc * (2 - params.cc) * params.mu_eff) * diff;
    %                                                           ↑ σ=1，直接乘diff

    % 论文公式3.4：rank-one更新
    params.C = (1 - params.c1) * params.C + ...
               params.c1 * (p_new * p_new');

    % 数值稳定
    params.C = (params.C + params.C') / 2;
    params.C = params.C + 1e-8 * eye(D);

    params.m = m_new;
    params.p = p_new;
end