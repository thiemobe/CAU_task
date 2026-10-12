%Comment lines 1-15
clear all
addpath("func\")

%CGD for coordinate descent, GD for gradient descent 
LfM = false; % LfM = true; %turn on&off learning from mistakes
epochs = 1000; % epochs to train
wDev = 0.1; % weights deviation
lrArr = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4 ];

load('MNISTdataset01.mat');
labelsTest = labelsTest+1;
labelsTrain = labelsTrain+1;

indLR = 0;
for lr = lrArr% learning rate 
    indLR = indLR + 1;

    trainingMode = 'GD';
    oneLayersGD_LfM
    
    save(['figures\Fig3_GDonEpochs_LR', num2str(lr), '.mat'])
    
    figure(1)
    subplot(length(lrArr), 2, 2*indLR-1)
    colorPl = [0.7 0 0];
    plot(accuracyTrain,'-', Color=colorPl)
    hold on
    plot(accuracyTest,'--',Color=colorPl)
    xlim ([0 epochs])
    ylim ([50 100])
    % legend('Training','Testing')
    % xlabel('Epochs')
    ylabel('Accuracy')
    box on
    
    % figure(3)
    subplot(length(lrArr), 2, 2*indLR)
    histogram(accuracyTrain(2:end),'Normalization', 'pdf','Orientation', 'horizontal', FaceColor=colorPl,BinWidth=1)
    ylim ([50 100])

    trainingMode = 'CD';
    oneLayersGD_LfM
    
    save(['figures\Fig3_CDonEpochs_LR', num2str(lr), '.mat'])
    
    figure(2)
    subplot(length(lrArr), 2, 2*indLR-1)
    colorPl = [0 0.5 0];
    plot(accuracyTrain,'-', Color=colorPl)
    hold on
    plot(accuracyTest,'--',Color=colorPl)
    xlim ([0 epochs])
    ylim ([50 100])
    % legend('Training','Testing')
    % xlabel('Epochs')
    ylabel('Accuracy')
    box on

    % figure(4)
    subplot(length(lrArr), 2, 2*indLR)
    histogram(accuracyTrain(2:end),'Normalization', 'pdf','Orientation', 'horizontal',FaceColor=colorPl,BinWidth=0.1)
    ylim ([50 100])



end

figure(1)
subplot(length(lrArr), 2, 2*indLR-1)
legend('Training','Testing')
xlabel('Epochs')

subplot(length(lrArr), 2, 2*indLR)
xlabel('Probability density')

figure(2)

subplot(length(lrArr), 2, 2*indLR-1)
legend('Training','Testing')
xlabel('Epochs')

subplot(length(lrArr), 2, 2*indLR)
xlabel('Probability density')

%% just plot
lrArr = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4 ];
indLR = 0;
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


for lr = lrArr% learning rate 
    indLR = indLR + 1;
    
    load(['figures\Fig3_GDonEpochs_LR', num2str(lr), '.mat'])
    
    figure(1)

    % s1 = subplot(length(lrArr), 2, 2*indLR-1)
    ax1(indLR) = nexttile;
    colorPl = [0.7 0 0];
    plot(accuracyTrain,'-', Color=colorPl)
    % hold on
    % plot(accuracyTest,'--',Color=colorPl)
    xlim ([0 epochs+100])
    ylim ([45 105])
    % legend('Training','Testing')
    % xlabel('Epochs')
    % ylabel('Accuracy ')
    box on
    if indLR == 7 
        xticks([0 epochs]); 
    else  
        xticks([]) 
    end
    yticks([50 100])


    % figure(3)
    % s2 = subplot(length(lrArr), 2, 2*indLR);
    ax1(indLR+1) = nexttile;

    histogram(accuracyTrain(2:end),'Normalization', 'pdf','Orientation', 'horizontal', FaceColor=colorPl,BinWidth=2)
    ylim ([45 105])
    xlim ([0 0.31])
    % accuracyMeanGD(indLR,1)=median(accuracyTrain(2:end));
    % accuracyMeanGD(indLR,2)=std(accuracyTrain(2:end));
    text(0.5, 0.2, [num2str(round(median(accuracyTrain(2:end)), 1)), ' ± ',num2str( round(std(accuracyTrain(2:end)), 1)), '%'] , 'FontSize', 12,'Units','normalized');
    
    
    
    if indLR == 7
        xticks([0 0.3]); 
    else  
        xticks([]) 
    end
    yticks([])


    %%CD
    load(['figures\Fig3_CDonEpochs_LR', num2str(lr), '.mat'])
    % linkaxes([s1 s2],'y') % share y-axis

    figure(2)
    ax2(indLR) = nexttile;

    % subplot(length(lrArr), 2, 2*indLR-1)
    colorPl = [0.5 0.5 0];
    plot(accuracyTrain,'-', Color=colorPl)
    % hold on
    % plot(accuracyTest,'--',Color=colorPl)
    xlim ([0 epochs+100])
    ylim ([94 101])
    % legend('Training','Testing')
    % xlabel('Epochs')
    % ylabel('Accuracy')
    box on
    if indLR == 7 
        xticks([0 epochs]); 
    else  
        xticks([]) 
    end    
    yticks([95 100])
    if indLR == 1
        ylim ([50 100])
        yticks([60 100])
    elseif indLR == 2
        ylim ([80 100])
        yticks([90 100])
    end

    % figure(4)
    ax2(indLR+1) = nexttile;

    % s = subplot(length(lrArr), 2, 2*indLR);
    % s.Position(1) = s.Position(1)-0.08;
    % s.Position(3) = s.Position(3)-0.16;
    % histogram(accuracyTrain(2:end),'Normalization', 'pdf','Orientation', 'horizontal',FaceColor=colorPl,BinWidth=1)
    
    % accuracyMeanCD(indLR,1)=median(accuracyTrain(2:end));
    % accuracyMeanCD(indLR,2)=std(accuracyTrain(2:end));

    if indLR == 1
        histogram(accuracyTrain(2:end),'Normalization', 'pdf','Orientation', 'horizontal',FaceColor=colorPl,BinWidth=1)

        ylim ([50 100])
    elseif indLR == 2
        histogram(accuracyTrain(2:end),'Normalization', 'pdf','Orientation', 'horizontal',FaceColor=colorPl,BinWidth=1)

        ylim ([80 100])
    else
        histogram(accuracyTrain(2:end),'Normalization', 'pdf','Orientation', 'horizontal',FaceColor=colorPl,BinWidth=0.1)

        ylim ([94 101])

    end    
        xlim ([0 4.2])

    yticks([])
    text(0.5, 0.2, [num2str( round(median(accuracyTrain(2:end)), 1)), ' ± ',num2str( round(std(accuracyTrain(2:end)), 1)), '%'], 'FontSize', 12,'Units','normalized');
    if indLR == 7 
        xticks([0 4]); 
    else  
        xticks([]) 
    end
end

figure(1)

fontsize(12,"points")
% ax1(7,1).XTick = [0 epochs];
% % % legend('Training','Testing')
% % % xlabel('Epochs')
% xticks(ax1(end), [0 0.2])

    
figure(2)
fontsize(12,"points")
% subplot(length(lrArr), 2, 1)
% ylim ([50 100])
% yticks([50 100])
% 
% subplot(length(lrArr), 2, 2)
% ylim ([50 100])
% 
% 
% subplot(length(lrArr), 2, 3)
% ylim ([80 100])
% yticks([80 100])
% 
% subplot(length(lrArr), 2, 4)
% ylim ([80 100])

% subplot(length(lrArr), 2, 2*indLR-1)
% % legend('Training','Testing')
% xlabel('Epochs')
% 
% subplot(length(lrArr), 2, 2*indLR)
% xlabel('Probability density')

