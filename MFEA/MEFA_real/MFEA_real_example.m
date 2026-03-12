%% 测试 MFEA
clear; clc; close all; rng(42);

D=5;

% 定义任务
Tasks{1}.name = 'Rastrigin';
Tasks{1}.dim = D;
Tasks{1}.lb = -50*ones(1,D);
Tasks{1}.ub = 50*ones(1,D);
Tasks{1}.func = @(x) 10*30 + sum(x.^2 - 10*cos(2*pi*x));

Tasks{2}.name = 'Ackley';
Tasks{2}.dim = D;
Tasks{2}.lb = -32 * ones(1, D);
Tasks{2}.ub = 32 * ones(1, D);
Tasks{2}.func = @(x) -20*exp(-0.2*sqrt(sum(x.^2)/30)) - exp(sum(cos(2*pi*x))/30) + 20 + exp(1);

% 参数
params.N = 100;
params.maxGen = 100;
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
legend({Tasks{1}.name, Tasks{2}.name});
xlabel('Generation'); ylabel('Fitness'); grid on;