function [bestX, bestF] = MFEA(Tasks, params)
% MFEA - Multi-Factorial Evolutionary Algorithm
% Implementation strictly following Gupta, Ong, Feng, IEEE Trans. Evolutionary Computation, 2016
% Key features:
%   - Implicit genetic transfer via assortative mating and vertical cultural transmission
%   - Individuals evaluated only on the task corresponding to their skill factor (after initialization)
%   - Scalar fitness based on intra-task ranking within skill factor subgroups
%   - (μ+λ)-selection using scalar fitness

    K      = numel(Tasks);                    % number of tasks
    D      = GetParam(params, 'dim',    30);  % dimensionality of search space
    N      = GetParam(params, 'N',     100);  % population size
    maxGen = GetParam(params, 'maxGen', 50);  % maximum number of generations
    rmp    = GetParam(params, 'rmp',   0.3);  % random mating probability
    pc     = GetParam(params, 'pc',    0.9);  % crossover probability
    pm     = GetParam(params, 'pm',    1/D);  % mutation probability
    eta_c  = GetParam(params, 'eta_c',  20);  % SBX distribution index
    eta_m  = GetParam(params, 'eta_m',  20);  % polynomial mutation distribution index

    % ── Initialization Phase ─────────────────────────────────────
    % Step 1: Generate initial population uniformly in [0,1]^D
    Pop = rand(D, N);

    % Step 2: Full factorial evaluation (all individuals evaluated on all tasks)
    FCostFull = zeros(K, N);
    for k = 1:K
        for i = 1:N
            FCostFull(k,i) = Tasks{k}(Pop(:,i));
        end
    end

    % Step 3: Assign initial skill factors based on best factorial rank
    Rank = zeros(K, N);
    for k = 1:K
        [~, ord] = sort(FCostFull(k,:));
        Rank(k, ord) = 1:N;
    end
    [~, Skill] = min(Rank, [], 1);   % skill factor τ_i = argmin_k {rank_k(i)}

    % Step 4: Extract single factorial cost corresponding to skill factor
    Cost = FCostFull(sub2ind([K N], Skill, 1:N));

    % Step 5: Record best solutions from initial full evaluation
    bestF = min(FCostFull, [], 2)';
    bestX = zeros(D, K);
    for k = 1:K
        [~, idx] = min(FCostFull(k,:));
        bestX(:,k) = Pop(:,idx);
    end

    % Release full cost matrix (memory optimization)
    clear FCostFull;

    % ── Generational Loop ────────────────────────────────────────
    for gen = 1:maxGen

        % ── Offspring Generation via Assortative Mating ──────────
        idx_perm = randperm(N);
        OffPop   = zeros(D, N);
        OffSkill = zeros(1, N);
        OffCost  = zeros(1, N);

        ci = 1;
        for i = 1:2:N-1
            p1 = idx_perm(i);
            p2 = idx_perm(i+1);
            s1 = Skill(p1);
            s2 = Skill(p2);

            % Assortative mating: crossover if same skill or with probability rmp
            if s1 == s2 || rand() < rmp
                [c1, c2] = SBX(Pop(:,p1), Pop(:,p2), D, pc, eta_c);
                sk1 = s1; sk2 = s2;
                if s1 ~= s2
                    % Vertical cultural transmission: random inheritance of skill factor
                    if rand() < 0.5, sk1 = s2; end
                    if rand() < 0.5, sk2 = s1; end
                end
            else
                % Unrelated mating: only mutation, skill factor inherited
                c1  = PolyMut(Pop(:,p1), D, pm, eta_m);
                c2  = PolyMut(Pop(:,p2), D, pm, eta_m);
                sk1 = s1;
                sk2 = s2;
            end

            % Offspring evaluated only on task corresponding to inherited skill factor
            if ci <= N
                OffPop(:,ci)  = c1;
                OffSkill(ci)  = sk1;
                OffCost(ci)   = Tasks{sk1}(c1);
                ci = ci + 1;
            end
            if ci <= N
                OffPop(:,ci)  = c2;
                OffSkill(ci)  = sk2;
                OffCost(ci)   = Tasks{sk2}(c2);
                ci = ci + 1;
            end
        end

        % Handle last individual if N is odd
        if mod(N,2) == 1 && ci <= N
            p = idx_perm(N);
            c = PolyMut(Pop(:,p), D, pm, eta_m);
            sk = Skill(p);
            OffPop(:,ci) = c;
            OffSkill(ci) = sk;
            OffCost(ci)  = Tasks{sk}(c);
        end

        % ── (μ+λ)-Selection based on Scalar Fitness ─────────────
        AllPop   = [Pop,   OffPop];
        AllCost  = [Cost,  OffCost];
        AllSkill = [Skill, OffSkill];

        % Compute scalar fitness via intra-task ranking
        ScalarFit = zeros(1, 2*N);
        for k = 1:K
            idx = find(AllSkill == k);
            if isempty(idx), continue; end
            [~, ord] = sort(AllCost(idx), 'ascend');
            ranks = zeros(1, length(idx));
            ranks(ord) = 1:length(idx);
            ScalarFit(idx) = 1 ./ ranks;   % scalar fitness φ_i = 1 / rank_τ_i(i)
        end

        % Elitist selection: top N individuals by descending scalar fitness
        [~, sel_idx] = sort(ScalarFit, 'descend');
        sel_idx = sel_idx(1:N);

        Pop   = AllPop(:, sel_idx);
        Cost  = AllCost(sel_idx);
        Skill = AllSkill(sel_idx);

        % ── Update best solutions per task ───────────────────────
        for k = 1:K
            idx = find(Skill == k);
            if ~isempty(idx)
                [v, ii] = min(Cost(idx));
                if v < bestF(k)
                    bestF(k)   = v;
                    bestX(:,k) = Pop(:, idx(ii));
                end
            end
        end
    end
end


% ── Simulated Binary Crossover (SBX) ─────────────────────────────
function [c1, c2] = SBX(p1, p2, D, pc, eta)
    c1 = p1; c2 = p2;
    if rand() >= pc, return; end
    for j = 1:D
        if abs(p1(j)-p2(j)) <= 1e-14, continue; end
        y1 = min(p1(j),p2(j)); y2 = max(p1(j),p2(j));
        bd = min(y1, 1-y2);
        beta = 1 + 2*bd/(y2-y1);
        alp  = 2 - beta^(-(eta+1));
        u = rand();
        if u <= 1/alp
            betaq = (u*alp)^(1/(eta+1));
        else
            betaq = (1/(2-u*alp))^(1/(eta+1));
        end
        half_s = 0.5*(y1+y2);
        half_d = 0.5*betaq*(y2-y1);
        c1(j) = max(0, min(1, half_s - half_d));
        c2(j) = max(0, min(1, half_s + half_d));
    end
end


% ── Polynomial Mutation ──────────────────────────────────────────
function c = PolyMut(p, D, pm, eta)
    c = p;
    for j = 1:D
        if rand() >= pm, continue; end
        x = p(j); u = rand();
        if u <= 0.5
            delta = (2*u + (1-2*u)*(1-x)^(eta+1))^(1/(eta+1)) - 1;
        else
            delta = 1 - (2*(1-u) + 2*(u-0.5)*x^(eta+1))^(1/(eta+1));
        end
        c(j) = max(0, min(1, x+delta));
    end
end


% ── Parameter retrieval with default ─────────────────────────────
function v = GetParam(s, f, d)
    if isfield(s,f), v = s.(f); else, v = d; end
end
