%%
clear all
addpath("func\")
% lrArr = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4 ];%v0
lrArr = [1e-2, 1e-1, 1e0, 1e1, 1e2 ];%v1

ms = 15;
load('figures\Fig4d_meanAcc.mat')
figure()
% colorPl = [0.7 0 0];
% errorbar(lrArr,accuracyMeanGD(:,1),accuracyMeanGD(:,2),'.', Color=colorPl)
% hold on
% colorPl = [0.5 0.5 0];
% errorbar(lrArr,accuracyMeanCD(:,1),accuracyMeanCD(:,2),'.', Color=colorPl)

colorPl = [1 0.5 0];
errorbar(lrArr,accuracyMeanGD_LfM(:,1),accuracyMeanGD_LfM(:,2),'.', Color=colorPl, MarkerSize=ms)
hold on
colorPl = [0 0.7 0];
errorbar(lrArr,accuracyMeanCD_LfM(:,1),accuracyMeanCD_LfM(:,2),'.', Color=colorPl, MarkerSize=ms)

xscale log
xlim ([0.8e-2 1.5e4])
ylim ([97.5 100.01])
xlabel('Learning rate')
ylabel('Accuracy [%]')


%%
figure()
ms = 15;
% colorPl = [0.7 0 0];
% plot(lrArr,accuracyMeanGD(:,1),'.', Color=colorPl, MarkerSize=ms)
% hold on
% colorPl = [0.5 0.5 0];
% plot(lrArr,accuracyMeanCD(:,1),'.', Color=colorPl, MarkerSize=ms)

colorPl = [1 0.5 0];
plot(lrArr,accuracyMeanGD_LfM(:,1),'.', Color=colorPl, MarkerSize=ms)

colorPl = [0 0.7 0];
plot(lrArr,accuracyMeanCD_LfM(:,1),'.', Color=colorPl, MarkerSize=ms)

xscale log

ylim ([75 100.01])
xlabel('Learning rate')
ylabel('Accuracy [%]')

