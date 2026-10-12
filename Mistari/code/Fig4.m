%Comment lines 1-15
clear all
addpath("func\")

%CGD for coordinate descent, GD for gradient descent 
LfM = true; %LfM = false; %  %turn on&off learning from mistakes
epochs = 1000; % epochs to train
wDev = 0.1; % weights deviation
% lrArr = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4 ];
lrArr = [1e-2, 1e-1, 1e0, 1e1, 1e2 ];%v1

load('MNISTdataset01.mat');
labelsTest = labelsTest+1;
labelsTrain = labelsTrain+1;

indLR = 0;
for lr = lrArr% learning rate 
    % indLR = indLR + 1;

    trainingMode = 'GD';
    oneLayersGD_LfM
    
    save(['figures\Fig4_GDonEpochs_LR', num2str(lr), '.mat'])
    
    % figure(1)
    % subplot(length(lrArr), 2, 2*indLR-1)
    % colorPl = [1 0.5 0];
    % plot(accuracyTrain,'-', Color=colorPl)
    % hold on
    % plot(accuracyTest,'--',Color=colorPl)
    % xlim ([0 epochs])
    % ylim ([50 100])
    % % legend('Training','Testing')
    % % xlabel('Epochs')
    % ylabel('Accuracy')
    % box on
    
    % figure(3)
    % subplot(length(lrArr), 2, 2*indLR)
    % histogram(accuracyTrain(2:end),'Normalization', 'pdf','Orientation', 'horizontal', FaceColor=colorPl,BinWidth=1)
    % ylim ([50 100])

    trainingMode = 'CD';
    oneLayersGD_LfM
    
    save(['figures\Fig4_CDonEpochs_LR', num2str(lr), '.mat'])
    
    % figure(2)
    % subplot(length(lrArr), 2, 2*indLR-1)
    % colorPl = [0 0.7 0];
    % plot(accuracyTrain,'-', Color=colorPl)
    % hold on
    % plot(accuracyTest,'--',Color=colorPl)
    % xlim ([0 epochs])
    % ylim ([50 100])
    % % legend('Training','Testing')
    % % xlabel('Epochs')
    % ylabel('Accuracy')
    % box on
    % 
    % % figure(4)
    % subplot(length(lrArr), 2, 2*indLR)
    % histogram(accuracyTrain(2:end),'Normalization', 'pdf','Orientation', 'horizontal',FaceColor=colorPl,BinWidth=0.1)
    % ylim ([50 100])



end
% 
% figure(1)
% subplot(length(lrArr), 2, 2*indLR-1)
% legend('Training','Testing')
% xlabel('Epochs')
% 
% subplot(length(lrArr), 2, 2*indLR)
% xlabel('Probability density')
% 
% figure(2)
% 
% subplot(length(lrArr), 2, 2*indLR-1)
% legend('Training','Testing')
% xlabel('Epochs')
% 
% subplot(length(lrArr), 2, 2*indLR)
% xlabel('Probability density')

%% just plot
lrArr = [1e-2, 1e-1, 1e0, 1e1];
figure(1)
hold off
figure(2)
hold off


figure(1)
% fontsize(12,"points")
t = tiledlayout(length(lrArr),2);
t.TileSpacing = 'none';
t.Padding = 'none';
ax1 = gobjects(length(lrArr),2);

figure(2)
% fontsize(12,"points")
t = tiledlayout(length(lrArr),2);
t.TileSpacing = 'none';
t.Padding = 'none';
ax2 = gobjects(length(lrArr),2);

figNum = length(lrArr);

indLR = 0;
for lr = lrArr% learning rate 
    indLR = indLR + 1;

    load(['figures\Fig4_GDonEpochs_LR', num2str(lr), '.mat'])

    figure(1)
    % subplot(figNum, 2, 2*indLR-1)
    ax1(indLR) = nexttile;

    colorPl = [1 0.5 0];
    plot(accuracyTrain,'-', Color=colorPl)
    % hold on
    % plot(accuracyTest,'--',Color=colorPl)
    xlim ([0 epochs+200])
    ylim ([50 100])
    % legend('Training','Testing')
    % xlabel('Epochs')
    % ylabel('Accuracy')
    box on

    if indLR == 4
        xticks([0 epochs]); 
    else  
        xticks([]) 
    end
    yticks([60 100]);
    % figure(3)
    % subplot(figNum, 2, 2*indLR)
    
    ax1(indLR) = nexttile;

    histogram(accuracyTrain(2:end),'Normalization', 'pdf','Orientation', 'horizontal', FaceColor=colorPl,BinWidth=1)
    ylim ([50 100])
    accuracyMeanGD_LfM(indLR,1)=median(accuracyTrain(2:end));
    accuracyMeanGD_LfM(indLR,2)=std(accuracyTrain(2:end));
    text(0.2, 0.2, [num2str( round(median(accuracyTrain(2:end)), 2)), ' ± ',num2str( round(std(accuracyTrain(2:end)), 1)), '%'], 'FontSize', 12,'Units','normalized');


    if indLR == 4 
        xticks([0 0.3]); 
    else  
        xticks([]) 
    end
    yticks([]); 

    load(['figures\Fig4_CDonEpochs_LR', num2str(lr), '.mat'])

    figure(2)
    % subplot(figNum, 2, 2*indLR-1)
    ax1(indLR+1) = nexttile;

    colorPl = [0 0.7 0];
    plot(accuracyTrain,'-', Color=colorPl)
    % hold on
    % plot(accuracyTest,'--',Color=colorPl)
    xlim ([0 epochs+200])
    ylim ([95 100])
    % legend('Training','Testing')
    % xlabel('Epochs')
    % ylabel('Accuracy')
    box on
    if indLR == 4 
        xticks([0 epochs]); 
    else  
        xticks([]) 
    end
    yticks([]); 

    % figure(4)
    % subplot(figNum, 2, 2*indLR)
    ax1(indLR+1) = nexttile;

    histogram(accuracyTrain(2:end),'Normalization', 'pdf','Orientation', 'horizontal',FaceColor=colorPl,BinWidth=0.1)
    xlim ([0 5])
    ylim ([95 100])
    accuracyMeanCD_LfM(indLR,1)=median(accuracyTrain(2:end));
    accuracyMeanCD_LfM(indLR,2)=std(accuracyTrain(2:end));
    text(0.2, 0.2, [num2str( round(median(accuracyTrain(2:end)), 1)), ' ± ',num2str( round(std(accuracyTrain(2:end)), 1)), '%'], 'FontSize', 12,'Units','normalized');
    
    if indLR == 4 
        xticks([0 4]); 
    else  
        xticks([]) 
    end
    yticks([96 100]); 
    set(ax1(indLR+1), 'YAxisLocation', 'right');

end


figure(1)
fontsize(12,"points")

    
figure(2)
fontsize(12,"points")

% figure(1)
% subplot(length(lrArr), 2, 2*indLR-1)
% % legend('Training','Testing')
% xlabel('Epochs')
% 
% subplot(length(lrArr), 2, 2*indLR)
% xlabel('Probability density')

% figure(2)
% subplot(length(lrArr), 2, 1)
% ylim ([50 100])
% subplot(length(lrArr), 2, 2)
% ylim ([50 100])

% subplot(length(lrArr), 2, 3)
% ylim ([80 100])
% subplot(length(lrArr), 2, 4)
% ylim ([80 100])

% subplot(length(lrArr), 2, 2*indLR-1)
% % legend('Training','Testing')
% xlabel('Epochs')
% 
% subplot(length(lrArr), 2, 2*indLR)
% xlabel('Probability density')



%%
% 
% 
% lrArr = [1e-2, 1e-1, 1e0, 1e1];
% figure(1)
% hold off
% figure(1)
% % fontsize(12,"points")
% t = tiledlayout(length(lrArr),2);
% t.TileSpacing = 'none';
% t.Padding = 'none';
% ax1 = gobjects(length(lrArr),2);
% 
% 
% figure(2)
% % fontsize(12,"points")
% t = tiledlayout(length(lrArr),2);
% t.TileSpacing = 'none';
% t.Padding = 'none';
% ax2 = gobjects(length(lrArr),2);
% 
% % figNum = length(lrArr);
% % t = tiledlayout(figNum, 2, 'TileSpacing', 'compact'); % 2 rows, 1 col
% indLR = 0;
% 
% 
% for lr = lrArr% learning rate 
%     indLR = indLR + 1;
% 
%     load(['figures\Fig4_GDonEpochs_LR', num2str(lr), '.mat'])
% 
%     figure(1)
%     ax1(indLR) = nexttile;
%     colorPl = [1 0.5 0];
%     plot(accuracyTrain,'-', Color=colorPl)
%     % hold on
%     % plot(accuracyTest,'--',Color=colorPl)
%     xlim ([0 epochs])
%     ylim ([50 100])
%     % legend('Training','Testing')
%     % xlabel('Epochs')
%     % ylabel('Accuracy')
%     box on
% 
%     figure(2)
%     ax2(indLR) = nexttile;
% 
%     histogram(accuracyTrain(2:end), 'Normalization', 'pdf', 'Orientation', 'horizontal', FaceColor = colorPl,BinWidth = 1)
%     ylim ([50 100])
%     accuracyMeanGD_LfM(indLR,1)=median(accuracyTrain(2:end));
%     accuracyMeanGD_LfM(indLR,2)=std(accuracyTrain(2:end));
%     text(0.2, 0.2, [num2str( round(median(accuracyTrain(2:end)), 2)), '±',num2str( round(std(accuracyTrain(2:end)), 1)), '%'], 'FontSize', 12,'Units','normalized');
% 
% end
