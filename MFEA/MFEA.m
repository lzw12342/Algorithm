function [bestX, bestF] = MFEA(Tasks, params)
% MFEA — Multifactorial Evolutionary Algorithm
%
% Faithful implementation of:
%   Gupta, Ong, Feng, IEEE Trans. Evol. Comput., 20(3):343-357, 2016.
%
% Performance optimisations (all paper-compliant, no parallelism):
%   1. AllFCost column-fill replaced by vectorised linear indexing
%   2. ComputeFitness inner loop vectorised: sort on full row at once,
%      +Inf entries naturally sort last so no find() needed
%   3. SBX inner loop replaced with fully vectorised operations
%   4. GaussMut vectorised (single randn call)
%   5. arrayfun at init replaced by direct linear indexing
%   6. fmincon opts struct built once outside all loops
%   7. Pre-allocated OffPop/OffSkill/OffCost outside the generation loop
%   8. AssignSkill uses vectorised sort + cumulative rank trick
%
% State per generation:
%   Pop   — D×N  (unified [0,1] random keys)
%   Skill — 1×N  skill factor τ_i
%   Cost  — 1×N  factorial cost on individual's skill-factor task

    K        = numel(Tasks);
    D        = GetParam(params, 'dim',      30);
    N        = GetParam(params, 'N',       100);
    maxGen   = GetParam(params, 'maxGen',  500);
    rmp      = GetParam(params, 'rmp',     0.3);
    pc       = GetParam(params, 'pc',      0.9);
    eta_c    = GetParam(params, 'eta_c',    20);
    sigma    = GetParam(params, 'sigma',   0.1);
    pm       = GetParam(params, 'pm',      1/D);
    useBFGS  = GetParam(params, 'useBFGS', true);
    bfgsIter = GetParam(params, 'bfgsIter',  5);

    % Build fmincon options once — avoids repeated struct construction
    if useBFGS
        bfgsOpts = optimoptions('fmincon', ...
            'Algorithm',              'interior-point', ...
            'MaxIterations',          bfgsIter,          ...
            'MaxFunctionEvaluations', bfgsIter * 10,     ...
            'OptimalityTolerance',    1e-6,              ...
            'StepTolerance',          1e-8,              ...
            'Display',                'off');
        lb = zeros(D, 1);
        ub = ones(D, 1);
    else
        bfgsOpts = [];
        lb = []; ub = [];
    end

    % ── Initialisation ────────────────────────────────────────────────────
    Pop = rand(D, N);

    % Full factorial evaluation (Algorithm 1, line 2)
    FCostInit = zeros(K, N);
    for k = 1:K
        for i = 1:N
            FCostInit(k,i) = Tasks{k}(Pop(:,i));
        end
    end

    % Skill factors (Definition 4) — vectorised rank computation
    Skill = AssignSkill(FCostInit, K, N);

    % Cost on each individual's skill-factor task
    % Vectorised: linear index into FCostInit directly
    Cost = FCostInit(sub2ind([K, N], Skill, 1:N));   % 1×N

    % Best found
    bestF = min(FCostInit, [], 2)';
    bestX = zeros(D, K);
    for k = 1:K
        [~, idx]   = min(FCostInit(k,:));
        bestX(:,k) = Pop(:,idx);
    end
    clear FCostInit;

    % Pre-allocate offspring arrays once — reused every generation
    OffPop   = zeros(D, N);
    OffSkill = zeros(1, N);
    OffCost  = zeros(1, N);

    % Pre-allocate merged pool arrays once
    AllPop   = zeros(D, 2*N);
    AllSkill = zeros(1, 2*N);
    AllCost  = zeros(1, 2*N);
    AllFCost = zeros(K, 2*N);

    % ── Generational loop ─────────────────────────────────────────────────
    for gen = 1:maxGen

        perm = randperm(N);
        ci   = 1;

        % ── Offspring generation (Algorithms 2 & 3) ───────────────────────
        for pi = 1 : 2 : N-1
            if ci > N, break; end
            p1 = perm(pi);   p2 = perm(pi+1);
            s1 = Skill(p1);  s2 = Skill(p2);

            if s1 == s2 || rand() < rmp
                [c1, c2] = SBX(Pop(:,p1), Pop(:,p2), pc, eta_c);
                if s1 == s2
                    sk1 = s1;  sk2 = s1;
                else
                    sk1 = s1 + (s2-s1)*(rand()<0.5);   % branchless skill pick
                    sk2 = s1 + (s2-s1)*(rand()<0.5);
                end
            else
                c1  = GaussMut(Pop(:,p1), D, pm, sigma);
                c2  = GaussMut(Pop(:,p2), D, pm, sigma);
                sk1 = s1;  sk2 = s2;
            end

            if ci <= N
                [c1, fc1]    = EvalBFGS(c1, Tasks{sk1}, useBFGS, bfgsOpts, lb, ub);
                OffPop(:,ci) = c1;  OffSkill(ci) = sk1;  OffCost(ci) = fc1;
                ci = ci + 1;
            end
            if ci <= N
                [c2, fc2]    = EvalBFGS(c2, Tasks{sk2}, useBFGS, bfgsOpts, lb, ub);
                OffPop(:,ci) = c2;  OffSkill(ci) = sk2;  OffCost(ci) = fc2;
                ci = ci + 1;
            end
        end

        % Odd-N tail
        if mod(N,2) == 1 && ci <= N
            p  = perm(N);  sk = Skill(p);
            c  = GaussMut(Pop(:,p), D, pm, sigma);
            [c, fc]      = EvalBFGS(c, Tasks{sk}, useBFGS, bfgsOpts, lb, ub);
            OffPop(:,ci) = c;  OffSkill(ci) = sk;  OffCost(ci) = fc;
        end

        % ── Merge (Algorithm 1, line iii) ────────────────────────────────
        AllPop(:,1:N)     = Pop;      AllPop(:,N+1:end)   = OffPop;
        AllSkill(1:N)     = Skill;    AllSkill(N+1:end)   = OffSkill;
        AllCost(1:N)      = Cost;     AllCost(N+1:end)    = OffCost;

        % Build K×2N factorial cost matrix via vectorised linear indexing
        % (replaces the for-loop over 2N individuals)
        AllFCost(:)       = inf;
        AllFCost(sub2ind([K, 2*N], AllSkill, 1:2*N)) = AllCost;

        % ── Scalar fitness + skill update (Algorithm 1, line iv) ─────────
        [NewSkill, ScalarFit] = ComputeFitness(AllFCost, K, 2*N);

        % ── Elitist selection (Algorithm 1, line v) ───────────────────────
        [~, ord] = sort(ScalarFit, 'descend');
        sel      = ord(1:N);

        Pop   = AllPop(:, sel);
        Skill = NewSkill(sel);
        Cost  = AllCost(sel);

        % ── Update best ───────────────────────────────────────────────────
        for k = 1:K
            idx = find(Skill == k);
            if ~isempty(idx)
                [v, ii] = min(Cost(idx));
                if v < bestF(k)
                    bestF(k)    = v;
                    bestX(:, k) = Pop(:, idx(ii));
                end
            end
        end

    end
end


% ═══════════════════════════════════════════════════════════════════════
%  AssignSkill — initial full-matrix rank assignment (Definitions 2 & 4)
%  Vectorised: sort each row, write ranks back in one scatter step.
% ═══════════════════════════════════════════════════════════════════════
function Skill = AssignSkill(FCost, K, N)
    Rank = zeros(K, N);
    for k = 1:K
        [~, ord]    = sort(FCost(k,:));
        Rank(k,ord) = 1:N;
    end
    [~, Skill] = min(Rank, [], 1);
end


% ═══════════════════════════════════════════════════════════════════════
%  ComputeFitness — Definitions 2, 3, 4  (called every generation)
%
%  Key optimisation: instead of find(isfinite(...)) per row, we sort the
%  full row including +Inf values.  MATLAB's sort places +Inf at the end,
%  so finite entries naturally rank 1..m and +Inf entries rank m+1..N.
%  We then clamp +Inf entries back to N (worst rank) with a single
%  min() call — no branch, no find().
% ═══════════════════════════════════════════════════════════════════════
function [Skill, ScalarFit] = ComputeFitness(AllFCost, K, N)
    FRank = zeros(K, N);

    for k = 1:K
        row = AllFCost(k,:);
        % Add tiny random noise for tie-breaking (Section II)
        % Only on finite entries to avoid nan from Inf+rand
        noise          = zeros(1, N);
        finite_mask    = isfinite(row);
        noise(finite_mask) = rand(1, sum(finite_mask)) * 1e-12;
        [~, ord]       = sort(row + noise);   % +Inf sorts to end automatically
        r              = zeros(1, N);
        r(ord)         = 1:N;
        % Clamp Inf-cost individuals to rank N (they sorted to positions
        % > number-of-finite entries, so their ranks are already >= that
        % count+1; force them all to exactly N for Definition 3 consistency)
        r(~finite_mask) = N;
        FRank(k,:)      = r;
    end

    [minRank, Skill] = min(FRank, [], 1);
    ScalarFit        = 1 ./ minRank;
end


% ═══════════════════════════════════════════════════════════════════════
%  EvalBFGS — evaluate then optional Lamarckian BFGS (Section V)
%  opts/lb/ub built once outside the loop and passed in.
% ═══════════════════════════════════════════════════════════════════════
function [x, fx] = EvalBFGS(x, taskFn, useBFGS, opts, lb, ub)
    fx = taskFn(x);
    if ~useBFGS, return; end
    try
        [x_opt, fx_opt, ef] = fmincon(taskFn, x, [], [], [], [], lb, ub, [], opts);
        if ef >= 0 && fx_opt < fx
            x  = x_opt;
            fx = fx_opt;
        end
    catch
    end
end


% ═══════════════════════════════════════════════════════════════════════
%  SBX — Simulated Binary Crossover, fully vectorised over D dimensions
%  No per-gene for-loop: all D genes processed with logical masks.
% ═══════════════════════════════════════════════════════════════════════
function [c1, c2] = SBX(p1, p2, pc, eta)
    c1 = p1;  c2 = p2;
    if rand() >= pc, return; end

    D    = numel(p1);
    diff = abs(p1 - p2);
    act  = diff > 1e-14;          % genes where crossover can happen
    if ~any(act), return; end

    y1 = min(p1, p2);             % D×1, element-wise
    y2 = max(p1, p2);

    bd    = min(y1, 1 - y2);
    beta  = 1 + 2 .* bd ./ (y2 - y1);
    alp   = 2 - beta .^ (-(eta+1));

    u     = rand(D, 1);
    % Compute betaq without a branch: use the two formulae weighted by mask
    lo    = u <= 1./alp;
    betaq = lo  .* (u .* alp) .^ (1/(eta+1)) + ...
            ~lo .* (1 ./ (2 - u .* alp)) .^ (1/(eta+1));

    half_s = 0.5*(y1 + y2);
    half_d = 0.5*betaq .* (y2 - y1);

    new1 = max(0, min(1, half_s - half_d));
    new2 = max(0, min(1, half_s + half_d));

    c1(act) = new1(act);
    c2(act) = new2(act);
end


% ═══════════════════════════════════════════════════════════════════════
%  GaussMut — Gaussian Mutation, vectorised (paper Section V)
% ═══════════════════════════════════════════════════════════════════════
function c = GaussMut(p, D, pm, sigma)
    mask    = rand(D, 1) < pm;
    c       = p;
    c(mask) = c(mask) + sigma * randn(sum(mask), 1);
    c       = max(0, min(1, c));
end


% ═══════════════════════════════════════════════════════════════════════
%  GetParam
% ═══════════════════════════════════════════════════════════════════════
function v = GetParam(s, f, d)
    if isfield(s, f), v = s.(f); else, v = d; end
end