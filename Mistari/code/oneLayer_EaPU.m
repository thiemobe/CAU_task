addpath("func\")

%% load dataset to test
load('MNISTdataset01.mat');
% load('MNISTdataset.mat');
labelsTest = labelsTest+1;
labelsTrain = labelsTrain+1;

%% set parameters
lr = 0.001; % learning rate 
epochs = 100; % epochs to train
wDev = 0.1; % weights deviation
deltaWth = 0.1;
%% Initializing size and arrays
Nin = size(imagesTrain,1);
Nout = size(yTrain,1);
nTest = size(imagesTest,2); 
nTrain = size(imagesTrain,2);
accuracyTrain =  zeros(1,epochs);
accuracyTest = zeros(1,epochs);
loss = zeros(1,epochs);
outputTrain = zeros(1,size(imagesTrain,2));

%% Initializing weights 
initVariation = sqrt(2/(Nin+Nout));
w1 = randn(Nout,Nin)*initVariation;

%% training

for k = 1:epochs %Outer epoch loop

    % Accuracy calculation for training and testing datasets
    [accuracyTrain(k),a2] = calcAccuracy(w1,imagesTrain,labelsTrain);
    [accuracyTest(k),~] = calcAccuracy(w1,imagesTest,labelsTest);
    
     %print the progress
    if mod(k,100)==0
        disp(['Training epoch ', num2str(k), ': accuracy ' ,num2str(accuracyTrain(k)),'%'])
    end
    % Cross-entropy loss calculation
    loss(k) = - mean(sum(yTrain .* log(a2), 1));
    
        for j = 1:nTrain
        
            %Feed forward
            a1 = imagesTrain(:,j);
            z2 = w1*a1;
            a2 = fa(z2);

            % check the answer for current pattern
            [~,outputTrain(j)] = max(a2);

            if (outputTrain(j) ~= labelsTrain(j))
                % EaPU gradient descent
                
                %random weight update 
                for ind1 = 1:Nout
                    for ind2 = 1:Nin
                        grad2 = (a2(ind1) - yTrain(ind1, j)) * a1(ind2);
                        deltaW = lr*grad2;
                        p = deltaW/deltaWth;  
                        if (deltaW > deltaWth) || ((deltaW < deltaWth) && (binornd(1, p, 1, 1) == 1))
                              
                            w1(ind1,ind2) = w1(ind1,ind2) - deltaW + wDev*randn(1);
                   
                            %keep weights in range [-1 1]
                            if w1(ind1,ind2) > 1
                                w1(ind1,ind2) = 1 - wDev*abs(randn(1));
                            elseif w1(ind1,ind2) < -1
                                w1(ind1,ind2) = - 1 + wDev*abs(randn(1));
                            end
                        end
                    end 
                end



            end
        end

    %suffle the trainin dataset
    [imagesTrain, yTrain, labelsTrain] = shuffle(imagesTrain, yTrain, labelsTrain); %Shuffles order of the images for next epoch
    
end

k = k+1;

%accuracy calculation for training and testing datasets
[accuracyTrain(k),a2] = calcAccuracy(w1,imagesTrain,labelsTrain);
[accuracyTest(k),~] = calcAccuracy(w1,imagesTest,labelsTest);

% Cross-entropy loss calculation
loss(k) = - mean(sum(yTrain .* log(a2), 1));

% save (['figures/СGDlr2e_0sigmaW', num2str(wDev), '_1000epochs.mat'])


% toc
%%

% disp('Training done!')
% figure(1); 
% a2 = fa(w1*imagesTest);
% [~,outputTest] = max(a2);

%%
% fprintf('Accuracy: ');
% fprintf('%f',sum(outputTest == labelsTest)/size(imagesTest,2)*100);
% disp(' %');
% 
% %%
% % a2 = fa(w1*imagesTest);
% % [~,outputTest] = max(a2);
% % 
% % 
% % fprintf('Accuracy: ');
% % fprintf('%f',sum(outputTest == labelsTest)/size(imagesTest,2)*100);
% % disp(' %');
%%
% figure(3)
% % subplot(2,2,1)
% C = confusionmat(labelsTrain,outputTrain);
% confusionchart(C)

% subplot(2,2,2)
% plot(loss)
% xlabel('Epochs')
% ylabel('Loss')
% 
% subplot(2,2,3)
% C = confusionmat(labelsTest,outputTest);
% confusionchart(C)
% 
% figure()
% % subplot(2,2,4)
% hold off
% plot(accuracyTrain)
% hold on
% plot(accuracyTest)
% legend('Train','Test')
% xlabel('Epochs')
% ylabel('Accuracy')

%%
% 
% figure(12)
% % % % % 
% % hold off
% switch m
%     case 10
%         colorPl = [0 0 1  ];
%     case 100
%         colorPl = [0 0.3 0.7  ];
%     case 1000
%         colorPl = [0 0.5 0.5  ];
%     case 10000
%         colorPl = [0 0.7 0.3  ];
%     case 60000
%         colorPl = [0 1 0  ];
%     case 1
%         colorPl = [0.3 0 0.7  ];
%     otherwise
%         colorPl = [0 0 1  ];
% % end
        colorPl = [ 0  0 0.2 ];
% % 
figure(15)
plot(accuracyTrain,'-', Color=colorPl)
hold on
plot(accuracyTest,'--',Color=colorPl)
% legend('Training','Testing')
xlabel('Epochs')

ylabel('Accuracy')
