function [bestX, bestF, history] = MFEA_real(Tasks, params)
%MFEA 多因子进化算法（对照 Gupta et al. 2016）
%
% 用法：
%   [bestX, bestF]          = MFEA_real(Tasks, params)   % 不记录 history
%   [bestX, bestF, history] = MFEA_real(Tasks, params)   % 记录每代最优值
%
% Tasks : 1×K cell，每个元素是结构体，包含字段：
%           .dim  — 决策变量维度
%           .lb   — 下界向量 (dim×1)
%           .ub   — 上界向量 (dim×1)
%           .func — 目标函数句柄 f(x)，x 已解码到实际搜索空间
%           .name — （可选）任务名称字符串
%
% params: 结构体，可选字段（缺省值见下）：
%           .N          种群大小                      (默认 100)
%           .maxGen     最大代数                      (默认 500)
%           .rmp        随机交配概率                  (默认 0.3)
%           .pc         SBX 触发概率                  (默认 0.9)
%           .pm         每维变异概率                  (默认 1/D)
%           .eta_c      SBX 分布指数                  (默认 20)
%           .eta_m      PM  分布指数                  (默认 20)
%           .bfgs_iter  BFGS 局部搜索最大迭代步数     (默认 5，0 表示不做局部搜索)
%           .bfgs_tol   BFGS 梯度收敛阈值             (默认 1e-6)
%           .verbose    是否打印进度                  (默认 true)

    recordHistory = (nargout == 3);

    K        = length(Tasks);
    N        = GetParam(params, 'N',         100);
    maxGen   = GetParam(params, 'maxGen',    500);
    rmp      = GetParam(params, 'rmp',       0.3);
    pc       = GetParam(params, 'pc',        0.9);
    eta_c    = GetParam(params, 'eta_c',     20);
    eta_m    = GetParam(params, 'eta_m',     20);
    bfgsIter = GetParam(params, 'bfgs_iter', 5);
    bfgsTol  = GetParam(params, 'bfgs_tol',  1e-6);
    verbose  = GetParam(params, 'verbose',   true);

    D       = max(cellfun(@(t) t.dim, Tasks));
    pm_rate = GetParam(params, 'pm', 1/D);

    % lb/range 预计算，避免循环内重复取结构体字段和做减法
    TaskLb    = cellfun(@(t) t.lb(:),           Tasks, 'UniformOutput', false);
    TaskRange = cellfun(@(t) t.ub(:) - t.lb(:), Tasks, 'UniformOutput', false);
    TaskDim   = cellfun(@(t) t.dim,             Tasks);

    % bfgsIter 判断提到循环外，用函数句柄分派，消除 2N*maxGen 次冗余判断
    if bfgsIter > 0
        EvalOffspring = @(chrom, k) EvalWithBfgs(chrom, Tasks{k}, TaskLb{k}, TaskRange{k}, bfgsIter, bfgsTol);
    else
        EvalOffspring = @(chrom, k) EvalDirect(chrom, Tasks{k}, TaskLb{k}, TaskRange{k});
    end

    %% 初始化种群（Algorithm 1, Step 1-3）
    Pop   = rand(D, N);
    FCost = EvaluateAll(Pop, Tasks, TaskLb, TaskRange, TaskDim, K, N);
    [~, ~, Skill] = CalcMetrics(FCost);

    % 从初始全量评估提取各任务初始最优，作为 history/bestF 基准
    bestF = inf(1, K);
    bestX = cell(1, K);
    for k = 1:K
        [bestF(k), idx] = min(FCost(k,:));
        bestX{k}        = TaskLb{k} + Pop(1:TaskDim(k), idx) .* TaskRange{k};
    end

    if recordHistory
        history.bestFitness = repmat(bestF, maxGen, 1);
    end

    %% 主循环（Algorithm 1, Step 4）
    for gen = 1:maxGen

        % 交配产生子代（Algorithm 2 + Algorithm 3）
        [OffPop, OffSkill] = Mating(Pop, Skill, N, D, rmp, pc, pm_rate, eta_c, eta_m);

        % 子代评估（+ 可选 Lamarckian BFGS）
        OffCost = inf(K, 2*N);
        for i = 1:2*N
            k = OffSkill(i);
            [OffPop(:,i), OffCost(k,i)] = EvalOffspring(OffPop(:,i), k);
        end

        % 环境选择（Algorithm 1, Step v）
        [Pop, FCost, Skill] = Selection(Pop, FCost, OffPop, OffCost, N);

        % 更新全局最优
        for k = 1:K
            [curBest, idx] = min(FCost(k,:));   % min 对 inf 安全
            if curBest < bestF(k)
                bestF(k) = curBest;
                bestX{k} = TaskLb{k} + Pop(1:TaskDim(k), idx) .* TaskRange{k};
            end
        end

        % 记录 history（recordHistory 在循环外已确定，此处无条件写入）
        if recordHistory
            history.bestFitness(gen, :) = bestF;
        end

        if verbose && mod(gen, 100) == 0
            fprintf('Gen %4d | ', gen);
            for k = 1:K
                fprintf('%s: %.2e | ', GetName(Tasks{k}, k), bestF(k));
            end
            fprintf('\n');
        end
    end

    if recordHistory
        history.finalFitness = bestF;
        % 截断到极小正数，避免调用方 semilogy 遇到 log(0)
        history.bestFitness  = max(history.bestFitness, 1e-16);
    end
end


%% =========================================================
%  EvalDirect / EvalWithBfgs：子代评估策略
%% =========================================================
function [chrom, fx] = EvalDirect(chrom, Task, lb, range)
    x  = lb + chrom(1:Task.dim) .* range;
    fx = Task.func(x);
end

function [chrom, fx] = EvalWithBfgs(chrom, Task, lb, range, maxIter, tol)
    x       = lb + chrom(1:Task.dim) .* range;
    [x, fx] = BfgsLocal(x, Task, lb, range, maxIter, tol);
    chrom(1:Task.dim) = max(0, min(1, (x - lb) ./ range));
end


%% =========================================================
%  BfgsLocal：轻量级有界 BFGS 局部搜索（Lamarckian learning）
%% =========================================================
function [xBest, fBest] = BfgsLocal(x0, Task, lb, range, maxIter, eps_g)
    ub    = lb + range;
    f     = Task.func;
    n     = Task.dim;
    In    = eye(n);
    eps_g2 = eps_g^2;                   % 平方阈值，配合 g'*g 避免 sqrt

    x     = max(lb, min(ub, x0));
    fx    = f(x);
    H     = In;
    xBest = x;
    fBest = fx;

    g  = NumericalGrad(f, x, fx, n);
    gg = g' * g;
    if gg < eps_g2, return; end

    for iter = 1:maxIter 
        d  = -H * g;
        dg = d' * g;

        if dg >= 0
            H  = In;
            d  = -g;
            dg = -gg;
        end

        alpha = WolfeLineSearch(f, x, fx, dg, d, lb, ub);

        xNew = max(lb, min(ub, x + alpha * d));
        fNew = f(xNew);
        gNew = NumericalGrad(f, xNew, fNew, n);

        s  = xNew - x;
        y  = gNew - g;
        sy = s' * y;
        if sy > 1e-10
            rho = 1 / sy;
            Hy  = H * y;
            H   = H - rho * (s * Hy' + Hy * s') + (rho^2 * (y' * Hy) + rho) * (s * s');
        end

        x  = xNew;
        fx = fNew;
        g  = gNew;
        gg = g' * g;

        if fx < fBest
            xBest = x;
            fBest = fx;
        end

        if gg < eps_g2,                                      break; end
        if abs(fx - fBest) < 1e-12 * (1 + abs(fBest)),      break; end
    end
end


%% ---------------------------------------------------------
%  NumericalGrad：前向差分数值梯度，原地扰动避免向量拷贝
%% ---------------------------------------------------------
function g = NumericalGrad(f, x, fx, n)
    h = max(1e-8, abs(x) * 1e-6);
    g = zeros(n, 1);
    for j = 1:n
        xj   = x(j);
        x(j) = xj + h(j);
        g(j) = (f(x) - fx) / h(j);
        x(j) = xj;
    end
end


%% ---------------------------------------------------------
%  WolfeLineSearch：回退式 Armijo 线搜索
%% ---------------------------------------------------------
function alpha = WolfeLineSearch(f, x, fx, dg, d, lb, ub)
    alpha   = 1.0;
    c1dg    = 1e-4 * dg;
    rho     = 0.5;
    maxBack = 20;

    for iBack = 1:maxBack 
        xNew = max(lb, min(ub, x + alpha * d));
        if f(xNew) <= fx + alpha * c1dg
            break;
        end
        alpha = alpha * rho;
    end
end


%% =========================================================
%  Mating：Assortative Mating + Vertical Cultural Transmission
%  Algorithm 2 & Algorithm 3
%% =========================================================
function [OffPop, OffSkill] = Mating(Pop, Skill, N, D, rmp, pc, pm_rate, eta_c, eta_m)
    OffPop   = zeros(D, 2*N);
    OffSkill = zeros(1, 2*N);

    parents    = randi(N, 2, N);
    cross_rand = rand(1, N);
    skill_rand = rand(1, 2*N);

    for i = 1:N
        p1 = parents(1,i);  p2 = parents(2,i);
        s1 = Skill(p1);     s2 = Skill(p2);
        c1_idx = 2*i-1;     c2_idx = 2*i;

        if (s1 == s2) || (cross_rand(i) < rmp)
            [c1, c2] = Sbx(Pop(:,p1), Pop(:,p2), D, pc, eta_c);
            % s1==s2 时无需随机，直接继承；否则按 50% 概率互换
            if s1 == s2
                sk1 = s1;  sk2 = s2;
            else
                sk1 = s1 + (s2 - s1) * (skill_rand(c1_idx) < 0.5);
                sk2 = s2 + (s1 - s2) * (skill_rand(c2_idx) < 0.5);
            end
        else
            c1  = Pm(Pop(:,p1), D, eta_m, pm_rate);
            c2  = Pm(Pop(:,p2), D, eta_m, pm_rate);
            sk1 = s1;
            sk2 = s2;
        end

        OffPop(:,c1_idx) = c1;  OffSkill(c1_idx) = sk1;
        OffPop(:,c2_idx) = c2;  OffSkill(c2_idx) = sk2;
    end
end


%% =========================================================
%  Sbx：Simulated Binary Crossover，搜索空间 [0,1]
%% =========================================================
function [c1, c2] = Sbx(p1, p2, D, pc, eta)
    c1 = p1;  c2 = p2;

    if rand() >= pc, return; end

    dim_rand  = rand(D, 1);
    u_rand    = rand(D, 1);
    exp1      = 1 / (eta + 1);
    neg_eta1  = -(eta + 1);             % 预计算 beta 的指数

    for j = 1:D
        diff = p1(j) - p2(j);
        if dim_rand(j) >= 0.5 || abs(diff) <= 1e-14
            continue;
        end

        if diff < 0
            y1 = p1(j);  y2 = p2(j);
        else
            y1 = p2(j);  y2 = p1(j);
        end

        % 展开 min(y1, 1-y2)：y1 在 [0,1] 内，1-y2 也是，取小者
        spread = y2 - y1;
        bd     = min(y1, 1 - y2);       % boundary distance
        beta   = 1 + 2 * bd / spread;
        alpha  = 2 - beta^neg_eta1;
        u      = u_rand(j);
        inv_a  = 1 / alpha;

        if u <= inv_a
            betaq = (u * alpha)^exp1;
        else
            betaq = (1 / (2 - u * alpha))^exp1;
        end

        half_sum  = 0.5 * (y1 + y2);
        half_diff = 0.5 * betaq * spread;
        c1(j) = max(0, min(1, half_sum - half_diff));
        c2(j) = max(0, min(1, half_sum + half_diff));
    end
end


%% =========================================================
%  Pm：Polynomial Mutation，搜索空间 [0,1]
%% =========================================================
function c = Pm(p, D, eta, pm_rate)
    c = p;

    dim_rand = rand(D, 1);
    u_rand   = rand(D, 1);
    exp1     = 1 / (eta + 1);
    eta1     = eta + 1;                 % 预计算，避免循环内重复加法

    for j = 1:D
        if dim_rand(j) >= pm_rate, continue; end

        x = p(j);
        u = u_rand(j);
        if u <= 0.5
            delta = (2*u + (1 - 2*u) * (1-x)^eta1)^exp1 - 1;
        else
            delta = 1 - (2*(1-u) + 2*(u-0.5) * x^eta1)^exp1;
        end

        c(j) = max(0, min(1, x + delta));
    end
end


%% =========================================================
%  EvaluateAll：初始化全量评估（Algorithm 1, Step 2）
%% =========================================================
function FCost = EvaluateAll(Pop, Tasks, TaskLb, TaskRange, TaskDim, K, N)
    FCost = inf(K, N);
    for k = 1:K
        func = Tasks{k}.func;
        lb   = TaskLb{k};
        rng  = TaskRange{k};
        dim  = TaskDim(k);
        for i = 1:N
            FCost(k,i) = func(lb + Pop(1:dim,i) .* rng);
        end
    end
end


%% =========================================================
%  CalcMetrics：Factorial Rank / Scalar Fitness / Skill Factor
%  Definition 2、3、4（Section II）
%  inf 个体赋予哨兵排名 N+1，不参与 skill factor 竞争
%% =========================================================
function [Rank, ScalarFit, Skill] = CalcMetrics(FCost)
    [K, N] = size(FCost);
    Rank   = repmat(N+1, K, N);
    for k = 1:K
        row  = FCost(k,:);
        vIdx = find(row < Inf);         % 一次 find 同时得到索引和数量
        if isempty(vIdx), continue; end
        [~, order]      = sort(row(vIdx), 'ascend');
        Rank(k, vIdx(order)) = 1:numel(vIdx);
    end
    [minRank, Skill] = min(Rank, [], 1);
    ScalarFit = 1 ./ minRank;
end


%% =========================================================
%  Selection：环境选择（Algorithm 1, Step v）
%% =========================================================
function [NewPop, NewCost, NewSkill] = Selection(Pop, Cost, OffPop, OffCost, N)
    AllPop  = [Pop,  OffPop];
    AllCost = [Cost, OffCost];
    [~, ScalarFit, AllSkill] = CalcMetrics(AllCost);
    [~, idx] = sort(ScalarFit, 'descend');
    idx      = idx(1:N);
    NewPop   = AllPop(:,  idx);
    NewCost  = AllCost(:, idx);
    NewSkill = AllSkill(idx);
end


%% =========================================================
%  工具函数
%% =========================================================
function v = GetParam(s, f, d)
    if isfield(s, f), v = s.(f); else, v = d; end
end

function n = GetName(t, i)
    if isfield(t, 'name'), n = t.name; else, n = sprintf('Task%d', i); end
end
