%% 测试 MFEA
clear; clc; close all; rng(42);

Dim=30;

Tasks{1}.name = 'Sphere';
Tasks{1}.dim = Dim;
Tasks{1}.lb = -50 * ones(1, Dim);
Tasks{1}.ub = 50 * ones(1, Dim);
Tasks{1}.func = @(x) sum(x.^2);

% 任务2: 30维 Rastrigin (多峰)
Tasks{2}.name = 'Rastrigin';
Tasks{2}.dim = Dim;
Tasks{2}.lb = -50 * ones(1, Dim);
Tasks{2}.ub = 50 * ones(1, Dim);
Tasks{2}.func = @(x) 10*30 + sum(x.^2 - 10*cos(2*pi*x));

% 任务3: Ackley
Tasks{3}.name = 'Ackley';
Tasks{3}.dim = Dim;
Tasks{3}.lb = -32 * ones(1, Dim);
Tasks{3}.ub = 32 * ones(1, Dim);
Tasks{3}.func = @(x) -20*exp(-0.2*sqrt(sum(x.^2)/30)) - exp(sum(cos(2*pi*x))/30) + 20 + exp(1);
% 参数
params.N = 50;
params.maxGen = 50;
params.rmp = 0.9;  % 关键参数：控制任务间基因流动
params.bfgsTol = 1e-2;
params.bfgsIter = 0;

% 运行
tic
[bestX, bestF, history] = MFEA_real(Tasks, params);
toc

% 绘图
figure;
semilogy(history.bestFitness, 'LineWidth', 2);
legend({Tasks{1}.name, Tasks{2}.name,Tasks{3}.name});
xlabel('Generation'); ylabel('Fitness'); grid on;
