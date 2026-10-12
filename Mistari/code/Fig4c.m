%%
clear all
addpath("func\")

%CGD for coordinate descent, GD for gradient descent 
LfM = true; %LfM = false; %  %turn on&off learning from mistakes
epochs = 10000; % epochs to train
wDev = 0.1; % weights deviation
lr = 1e-1;

% lrArr = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4 ];%v0
lrArr = [1e-2, 1e-1, 1e0, 1e1, 1e2 ];%v1


trainingMode = 'GD';
    oneLayersGD_LfM
    save(['figures\Fig4c_GDonEpochs_LR', num2str(lr), '.mat'])
% 
% figure(5)
%     colorPl = [1 0.5 0];
%     plot(accuracyTrain,'-', Color=colorPl)

lr = 1e0;


    trainingMode = 'CD';
    oneLayersGD_LfM
    save(['figures\Fig4c_CDonEpochs_LR', num2str(lr), '.mat'])


% figure(5)
% hold on
% colorPl = [0 0.7 0];
% plot(accuracyTrain,'-', Color=colorPl)
% xlabel('Epochs')
% ylabel('Accuracy [%]')


%% 
figure(5)
lr = 0.1;
    load(['figures\Fig4c_GDonEpochs_LR', num2str(lr), '.mat'])
    colorPl = [1 0.5 0];
    plot(accuracyTrain,'-', Color=colorPl)
lr = 1;

    hold on
    load(['figures\Fig4c_CDonEpochs_LR', num2str(lr), '.mat'])
        colorPl = [0 0.7 0];
    plot(accuracyTrain,'-', Color=colorPl)
xlim ([1 10000]) 
ylim ([98 100.01])
xlabel('Epochs')
ylabel('Accuracy [%]')


