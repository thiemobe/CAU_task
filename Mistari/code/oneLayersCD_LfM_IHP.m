addpath("func\")
addpath("IHPfunctions\")
% add deviceWeightMapping,   create readState 

%% load dataset to test
load('MNISTdataset01.mat');
% load('MNISTdataset.mat');

labelsTest = labelsTest+1;
labelsTrain = labelsTrain+1;

%% set network parameters
LfM = true; % LfM = true; %turn on&off learning from mistakes
lr = 1; % learning rate 
epochs = 10000; % epochs to train

%% Initializing parameters 
% initVariation = sqrt(2/(Nin+Nout));
% w1 = randn(Nout,Nin)*initVariation;

%parameters for initialization
voltageToLRS = 1.5; 
voltageToHRS = 1.5; 
% voltageResetWin = 0.8;
% voltageSetWr = 0.5;
voltageRead = 0.2;
timestep = 1e-1;
IccSet = 1e-4;
IccReset = 2e-4;
Gmax = 180e-6;

%% Initializing size and arrays

Nin = size(imagesTrain,1);
Nout = size(yTrain,1);
nTest = size(imagesTest,2); 
nTrain = size(imagesTrain,2);
accuracyTrain = zeros(1,epochs);
accuracyTest = zeros(1,epochs);
loss = zeros(1,epochs);
outputTrain = zeros(1,size(imagesTrain,2));

updates = [];
% wArr = zeros(Nout,Nin,epochs);
% wPArr = zeros(Nout,Nin,epochs);
% wNegArr = zeros(Nout,Nin,epochs);
% GPosArr = zeros(Nout,Nin,epochs);
% GNegArr = zeros(Nout,Nin,epochs);
% 
% wArr = zeros(Nout,Nin,nTrain,epochs);
% wPosArr = zeros(Nout,Nin,nTrain,epochs);
% wNegArr = zeros(Nout,Nin,nTrain,epochs);
% GPosArr = zeros(Nout,Nin,nTrain,epochs);
% GNegArr = zeros(Nout,Nin,nTrain,epochs);
%% load map of working devices
if exist("crossbarMap.mat")
    load("crossbarMap.mat")
else
    disp('Create croosbar map with meas_all_IV.m')
end

%% Connect arduino and smu
if ~exist('arduinoObj')
    arduinoObj = connectArduino;
end

if ~exist('smu')
    smu = connectSMU;
end

%% weights initialization
% initVariation = sqrt(2/(Nin+Nout));
% w1 = randn(Nout,Nin)*initVariation;

%weights initialization
[devIndWpos,devRow,devCol] = deviceWeightMapping(HRS2LRS, Nout, Nin, 1, 1);
[devIndWneg,devRow,devCol] = deviceWeightMapping(HRS2LRS, Nout, Nin, devRow, devCol);

turnOnSMUChannels(smu)

%read the conductance Gpos for all devices in reservoir for different Vset
% Gpos = readMatrixState (arduinoObj, smu, devIndWpos, timestep, IccSet, voltageRead);
% Gneg = readMatrixState (arduinoObj, smu, devIndWneg, timestep, IccSet, voltageRead);

Gpos = resetMatrixState (arduinoObj, smu, devIndWpos, voltageToHRS,timestep, IccReset, voltageRead);
Gneg = resetMatrixState (arduinoObj, smu, devIndWneg, voltageToHRS, timestep, IccReset, voltageRead);
file='Gpos_init'; 
writematrix(Gpos,[file,'.txt']);

Gneg = readMatrixState (arduinoObj, smu, devIndWneg, timestep, IccSet, voltageRead);
file='Gneg_init'; 
writematrix(Gneg,[file,'.txt']);

w1pos = Gpos./Gmax;
w1neg = Gneg./Gmax;
w1 = w1pos-w1neg;


%% training

for k = 1:epochs %Outer epoch loop
    

    % Accuracy calculation for training and testing datasets
    [accuracyTrain(k),a2] = calcAccuracy(w1,imagesTrain,labelsTrain);
    [accuracyTest(k),~] = calcAccuracy(w1,imagesTest,labelsTest);

    %print the progress
    disp(['Training epoch ', num2str(k), ': accuracy ' ,num2str(accuracyTrain(k)),'%'])

    % Cross-entropy loss calculation
    loss(k) = - mean(sum(yTrain .* log(a2), 1));
    
    for j = 1:nTrain

            %Feed forward
            a1 = imagesTrain(:,j);
            z2 = w1*a1;
            a2 = fa(z2);

            % check the answer for current pattern
            [~,outputTrain(j)] = max(a2);

            if LfM
                condition = (outputTrain(j) ~= labelsTrain(j));
            else
                condition = true;
            end

            if condition
                    % Coordinate gradient descent
                    % choose random weight
                    ind1 = randi([1,Nout],1);
                    ind2 = randi([1,Nin],1);
    
                    %gradient calculation
                    grad2 = (a2(ind1) - yTrain(ind1, j)) * a1(ind2);
    
                    %random weight update 
                    deltaW = lr*grad2;
                    wUpdated = w1(ind1,ind2) - deltaW;
                    voltageToApply = volt(abs(wUpdated));

                    if sign(wUpdated)>0
                        setDeviceN(arduinoObj, devIndWpos(1, ind1, ind2), devIndWpos(2, ind1, ind2))
                        [~,~,~,~] = applyVoltage(smu, voltageToLRS, timestep, IccSet);
                        resetDeviceN(arduinoObj, devIndWpos(1, ind1, ind2), devIndWpos(2, ind1, ind2))
                        [~,~,~,~] = applyVoltage(smu, voltageToApply, timestep, IccReset);
                        setDeviceN(arduinoObj, devIndWpos(1, ind1, ind2), devIndWpos(2, ind1, ind2))
                        [~,~,R1,~] = applyVoltage(smu, voltageRead, timestep, IccSet);
                        Gpos(ind1,ind2) = 1/R1;
                        w1pos(ind1,ind2) = Gpos(ind1,ind2)/Gmax;

                    elseif sign(wUpdated)<0
                        setDeviceN(arduinoObj, devIndWneg(1, ind1, ind2), devIndWneg(2, ind1, ind2))
                        [~,~,~,~] = applyVoltage(smu, voltageToLRS, timestep, IccSet);
                        resetDeviceN(arduinoObj, devIndWneg(1, ind1, ind2), devIndWneg(2, ind1, ind2))
                        [~,~,~,~] = applyVoltage(smu, voltageToApply, timestep, IccReset);
                        setDeviceN(arduinoObj, devIndWneg(1, ind1, ind2), devIndWneg(2, ind1, ind2))
                        [~,~,R1,~] = applyVoltage(smu, voltageRead, timestep, IccSet);
                        Gneg(ind1,ind2) = 1/R1;
                        w1neg(ind1,ind2) = Gneg(ind1,ind2)/Gmax;
                    end
                    
                    if sign(wUpdated)<sign(w1(ind1,ind2))
                        resetDeviceN(arduinoObj, devIndWpos(1, ind1, ind2), devIndWpos(2, ind1, ind2))
                        [~,~,~,~] = applyVoltage(smu, voltageToHRS, timestep, IccReset);
                        setDeviceN(arduinoObj, devIndWpos(1, ind1, ind2), devIndWpos(2, ind1, ind2))
                        [~,~,R1,~] = applyVoltage(smu, voltageRead, timestep, IccSet);
                        Gpos(ind1,ind2) = 1/R1;
                        w1pos(ind1,ind2) = Gpos(ind1,ind2)/Gmax;
            
                    elseif sign(wUpdated)>sign(w1(ind1,ind2))
                        resetDeviceN(arduinoObj, devIndWneg(1, ind1, ind2), devIndWneg(2, ind1, ind2))
                        [~,~,~,~] = applyVoltage(smu, voltageToHRS, timestep, IccReset);
                        setDeviceN(arduinoObj, devIndWneg(1, ind1, ind2), devIndWneg(2, ind1, ind2))
                        [~,~,R1,~] = applyVoltage(smu, voltageRead, timestep, IccSet);
                        Gneg(ind1,ind2) = 1/R1;
                        w1neg(ind1,ind2) = Gneg(ind1,ind2)/Gmax;
                    end
                    w1 = w1pos-w1neg;
                    updates = [updates; j k Gpos(ind1,ind2) Gneg(ind1,ind2) ind1 ind2];       

            end
            % wPosArr (:,:,j,k) = w1pos;
            % wNegArr(:,:,j,k) = w1neg;
            % GPosArr(:,:,j,k) = Gpos;
            % GNegArr(:,:,j,k) = Gneg;
    end
    %suffle the trainin dataset
    [imagesTrain, yTrain, labelsTrain] = shuffle(imagesTrain, yTrain, labelsTrain); %Shuffles order of the images for next epoch
    save('CD_LfM_IHP.mat')
end

k = k+1;

%accuracy calculation for training and testing datasets
[accuracyTrain(k),a2] = calcAccuracy(w1,imagesTrain,labelsTrain);
[accuracyTest(k),~] = calcAccuracy(w1,imagesTest,labelsTest);

% Cross-entropy loss calculation
loss(k) = - mean(sum(yTrain .* log(a2), 1));

% save (['figures/СGDlr2e_0sigmaW', num2str(wDev), '_1000epochs.mat'])
turnOffSMUChannels(smu)


% toc
%%

% 
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
% % figure(10)
% subplot(2,2,4)
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
% end
        colorPl = [0 0 0 ];
% % 
figure(15)
plot(accuracyTrain,'-', Color=colorPl)
hold on
plot(accuracyTest,'--',Color=colorPl)
legend('Training','Testing')
xlabel('Epochs')

ylabel('Accuracy [%]')
