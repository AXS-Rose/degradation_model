%% Carga de datos
C1 = load("C:\Users\Bruno\OneDrive - Universidad de Chile" + ...
               "\BGMG\CASE\Datasets\Dataset Stanford\W10\C1.mat");
C2 = load("C:\Users\Bruno\OneDrive - Universidad de Chile" + ...
               "\BGMG\CASE\Datasets\Dataset Stanford\W10\C2.mat");
C3 = load("C:\Users\Bruno\OneDrive - Universidad de Chile" + ...
               "\BGMG\CASE\Datasets\Dataset Stanford\W10\C3.mat");
C4 = load("C:\Users\Bruno\OneDrive - Universidad de Chile" + ...
               "\BGMG\CASE\Datasets\Dataset Stanford\W10\C4.mat");
%% CON PROBLEMAS
C5 = load("C:\Users\Bruno\OneDrive - Universidad de Chile" + ...
               "\BGMG\CASE\Datasets\Dataset Stanford\W10\C5.mat");
%%
C6 = load("C:\Users\Bruno\OneDrive - Universidad de Chile" + ...
               "\BGMG\CASE\Datasets\Dataset Stanford\W10\C6.mat");
%% CON PROBLEMAS
C7 = load("C:\Users\Bruno\OneDrive - Universidad de Chile" + ...
               "\BGMG\CASE\Datasets\Dataset Stanford\W10\C7.mat");
%%
C8 = load("C:\Users\Bruno\OneDrive - Universidad de Chile" + ...
               "\BGMG\CASE\Datasets\Dataset Stanford\W10\C8.mat");
C9 = load("C:\Users\Bruno\OneDrive - Universidad de Chile" + ...
               "\BGMG\CASE\Datasets\Dataset Stanford\W10\C9.mat");
C10 = load("C:\Users\Bruno\OneDrive - Universidad de Chile" + ...
               "\BGMG\CASE\Datasets\Dataset Stanford\W10\C10.mat");
C11 = load("C:\Users\Bruno\OneDrive - Universidad de Chile" + ...
               "\BGMG\CASE\Datasets\Dataset Stanford\W10\C11.mat");
C12 = load("C:\Users\Bruno\OneDrive - Universidad de Chile" + ...
               "\BGMG\CASE\Datasets\Dataset Stanford\W10\C12.mat");
C13 = load("C:\Users\Bruno\OneDrive - Universidad de Chile" + ...
               "\BGMG\CASE\Datasets\Dataset Stanford\W10\C13.mat");
C14 = load("C:\Users\Bruno\OneDrive - Universidad de Chile" + ...
               "\BGMG\CASE\Datasets\Dataset Stanford\W10\C14.mat");

%%


datasets = [C1,C2,C3,C4,C6,C8,C9,C10,C11,C12,C13,C14];

% cnt = 0;
% for i = 1:length(datasets)
%     cnt = cnt + length(datasets(i).I_full_vec_M1_NMC25degC);
% end

I = [];
V = [];
time = [];
step = [];

for i = 1:length(datasets)
    I = [I;datasets(i).I_full_vec_M1_NMC25degC];
    V = [V;datasets(i).V_full_vec_M1_NMC25degC];
    time = [time;datasets(i).t_full_vec_M1_NMC25degC];
    step = [step;datasets(i).Step_Index_full_vec_M1_NMC25degC];
end

data = [I V time step];
data = array2table(data,VariableNames={'I','V','time','step'});
%%
dtime = zeros([length(I),1]);
%% conteo de coulomb

for i = 2:length(I)
    dt = time(i) - time(i-1);
    dtime(i) = dt;
end

%% conteo de coulomb
dq = zeros([length(I),1]);
dq_acc = zeros([length(I),1]);

for i = 2:length(I)
    dq(i) = (dtime(i) * data.I(i))/(4.865*3600);
    dq_acc(i) = dq_acc(i-1) + dq(i);
    if dq_acc(i) > 1
        dq_acc(i) = 1;
    elseif dq_acc(i) < 0
        dq_acc(i) = 0;
    else
    end
end

% for i = 2:length(I)
%     dq(i) = (dtime(i) * data.I(i))/(4.865*3600);
%     dq_acc(i) = dq_acc(i-1) + dq(i);
% end

%%
% Crea una figura
figure('Color', 'w');
% plotStyle = ['*'];
plot(dq_acc,'LineWidth', 1.5);
title('Conteo de Coulomb Dataset Stanford', 'FontSize', 14);
xlabel('Index', 'FontSize', 12);
ylabel('SoC (%)', 'FontSize', 12);
grid on;
%%
I_ = data.I(step == 7);


% plotStyle = ['*'];
unique_steps = unique(data.step);
for i = 1:length(unique_steps)
    figure('Color', 'w');
    plot(data.I(step == unique_steps(i)),'LineWidth', 1.5);
    title('Step: ',num2str(unique_steps(i)))
end
grid on;

%%
q_13 = 0;
q_14 = 0;
q_1 = 0;
for i = 1:length(I)
    if step(i) == 13
        q_13 = q_13 + dq(i);
    elseif step(i) == 14
        q_14 = q_14 + dq(i);
    else
        q_1 = q_1 + dq(i);
    end
end



