function oneLayer_CAU_Regression_1Device_IHP_Light( ...
    trainSamplesPerDigit, testSamplesPerDigit, epochs)
% Light hardware smoke test for 2x2 digit-to-letter regression.
%setenv('CAU_RUN_HARDWARE', '1')
%oneLayer_CAU_Regression_1Device_IHP_Light(100, 10, 1)

if nargin < 1 || isempty(trainSamplesPerDigit)
    trainSamplesPerDigit = 100;
end
if nargin < 2 || isempty(testSamplesPerDigit)
    testSamplesPerDigit = 10;
end
if nargin < 3 || isempty(epochs)
    epochs = 1;
end
validateattributes(trainSamplesPerDigit, {'numeric'}, ...
    {'scalar', 'integer', 'positive'});
validateattributes(testSamplesPerDigit, {'numeric'}, ...
    {'scalar', 'integer', 'positive'});
validateattributes(epochs, {'numeric'}, {'scalar', 'integer', 'positive'});

rng(123);
scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));
funcDir = fullfile(scriptDir, 'func');
ihpFunctionsDirs = {fullfile(fileparts(scriptDir), 'IHPfunctions'), ...
    fullfile(scriptDir, 'IHPfunctions'), fullfile(projectRoot, 'IHPfunctions')};
if isfolder(scriptDir)
    addpath(scriptDir, '-begin');
end
if isfolder(funcDir)
    addpath(funcDir);
end
for pathIndex = 1:numel(ihpFunctionsDirs)
    if isfolder(ihpFunctionsDirs{pathIndex})
        addpath(ihpFunctionsDirs{pathIndex});
    end
end

inputFile = fullfile(funcDir, 'MNISTdataset0123.mat');
if ~isfile(inputFile)
    inputFile = fullfile(projectRoot, 'Inputs', 'MNISTdataset0123.mat');
end
outputsDir = fullfile(scriptDir, 'Outputs');
if ~isfolder(outputsDir) && isfolder(fullfile(projectRoot, 'Outputs'))
    outputsDir = fullfile(projectRoot, 'Outputs');
end
resultsDir = fullfile(outputsDir, sprintf( ...
    'hardware_regression_1device_light_train%d_test%d', ...
    trainSamplesPerDigit, testSamplesPerDigit));

inputSize = 2;
outputSize = 2;
learningRate = 1.0;
sampleMSEThreshold = 0.2;
weightLimit = 1.0;
gVoltageScale = 160e-6;
gMin = 3.72159e-6;
gMax = 1.38826e-4;
gBias = (gMin + gMax) / 2;
gHalfRange = (gMax - gMin) / 2;
voltageToLRS = 1.5;
voltageRead = 0.2;
timestep = 1e-1;
iccSet = 1e-4;
iccReset = 2e-4;
iccRead = 5e-5;
postPulseReadDelay = 1.0;
readbackMaxAttempts = 3;
readbackRetryDelay = 1.0;
readCurrentComplianceFraction = 0.95;
readVoltageRelativeTolerance = 0.05;
readVoltageAbsoluteTolerance = 5e-3;
digitLabels = [1, 2, 3];
letterNames = {'c', 'a', 'u'};
runHardware = strcmp(getenv('CAU_RUN_HARDWARE'), '1');

if ~isfile(inputFile)
    error('MNIST input file not found: %s', inputFile);
end
matData = load(inputFile, 'imagesTrain', 'labelsTrain', 'imagesTest', 'labelsTest');
requiredFields = {'imagesTrain', 'labelsTrain', 'imagesTest', 'labelsTest'};
if ~all(isfield(matData, requiredFields))
    error('MAT file must contain imagesTrain, labelsTrain, imagesTest, and labelsTest.');
end
imagesTrain = double(matData.imagesTrain);
labelsTrain = double(matData.labelsTrain(:));
imagesTest = double(matData.imagesTest);
labelsTest = double(matData.labelsTest(:));
validateImageMatrix(imagesTrain, 'imagesTrain');
validateImageMatrix(imagesTest, 'imagesTest');
if size(imagesTrain, 1) ~= size(imagesTest, 1)
    error('Training and test images must have matching feature counts.');
end

targetByDigit = zeros(outputSize^2, numel(digitLabels));
for digitPosition = 1:numel(digitLabels)
    targetPath = findLetterTarget(outputsDir, letterNames{digitPosition}, outputSize);
    targetImage = imread(targetPath);
    if ndims(targetImage) == 3
        targetImage = rgb2gray(targetImage(:, :, 1:3));
    end
    targetImage = im2double(imresize(targetImage, ...
        [outputSize, outputSize], 'bilinear'));
    targetByDigit(:, digitPosition) = targetImage(:);
end

[inputBatch, targetBatch, trainSampleIndices, trainDigitLabels] = ...
    makeBatch(imagesTrain, labelsTrain, trainSamplesPerDigit, ...
    digitLabels, targetByDigit, inputSize);
[testInputBatch, testTargetBatch, testSampleIndices, testDigitLabels] = ...
    makeBatch(imagesTest, labelsTest, testSamplesPerDigit, ...
    digitLabels, targetByDigit, inputSize);

Nin = size(inputBatch, 1);
Nout = size(targetBatch, 1);
requiredDevices = Nin * Nout;
fprintf('Light CAU hardware run: %d x %d input/output pixels, %d devices.\n', ...
    inputSize, outputSize, requiredDevices);
fprintf('Training: %d (%d/%d/%d); test: %d (%d/%d/%d).\n', ...
    size(inputBatch, 2), sum(trainDigitLabels == 1), ...
    sum(trainDigitLabels == 2), sum(trainDigitLabels == 3), ...
    size(testInputBatch, 2), sum(testDigitLabels == 1), ...
    sum(testDigitLabels == 2), sum(testDigitLabels == 3));
fprintf('Epochs: %d; sample MSE threshold: %.4g.\n', ...
    epochs, sampleMSEThreshold);

mapPath = '';
mapCandidates = cellfun(@(folder) fullfile(folder, 'crossbarMap.mat'), ...
    ihpFunctionsDirs, 'UniformOutput', false);
mapCandidates{end + 1} = fullfile(scriptDir, 'crossbarMap.mat');
mapCandidates{end + 1} = fullfile(pwd, 'crossbarMap.mat');
for pathIndex = 1:numel(mapCandidates)
    if isfile(mapCandidates{pathIndex})
        mapPath = mapCandidates{pathIndex};
        break;
    end
end
if isfile(mapPath)
    mapData = load(mapPath, 'HRS2LRS');
    if ~isfield(mapData, 'HRS2LRS')
        error('crossbarMap.mat must contain HRS2LRS.');
    end
    availableDevices = numel(mapData.HRS2LRS);
    fprintf('Crossbar capacity: %d sites; current map flags %d.\n', ...
        availableDevices, nnz(mapData.HRS2LRS));
else
    mapData = struct();
    availableDevices = NaN;
    fprintf('crossbarMap.mat not found; map capacity cannot be checked yet.\n');
end

if ~runHardware
    fprintf(['Hardware is disabled. Review setup, then setenv(''CAU_RUN_HARDWARE'', ', ...
        '''1'') and call this function again to run on the array.\n']);
    return;
end
if ~isfile(mapPath)
    error('crossbarMap.mat not found. Generate the hardware map before running.');
end
if requiredDevices > availableDevices
    error('This model needs %d sites, but the crossbar has only %d.', ...
        requiredDevices, availableDevices);
end

config = struct( ...
    'inputSize', inputSize, 'outputSize', outputSize, 'epochs', epochs, ...
    'trainSamplesPerDigit', trainSamplesPerDigit, ...
    'testSamplesPerDigit', testSamplesPerDigit, ...
    'learningRate', learningRate, 'sampleMSEThreshold', sampleMSEThreshold, ...
    'gMin', gMin, 'gMax', gMax, 'gVoltageScale', gVoltageScale, ...
    'gBias', gBias, 'gHalfRange', gHalfRange, 'weightLimit', weightLimit, ...
    'voltageToLRS', voltageToLRS, 'voltageRead', voltageRead, ...
    'timestep', timestep, 'iccSet', iccSet, 'iccReset', iccReset, ...
    'iccRead', iccRead, 'postPulseReadDelay', postPulseReadDelay, ...
    'readbackMaxAttempts', readbackMaxAttempts, ...
    'readbackRetryDelay', readbackRetryDelay, ...
    'iccBias', min(iccSet, iccReset), ...
    'readCurrentComplianceFraction', readCurrentComplianceFraction, ...
    'readVoltageRelativeTolerance', readVoltageRelativeTolerance, ...
    'readVoltageAbsoluteTolerance', readVoltageAbsoluteTolerance);

runHardwareTraining(inputBatch, targetBatch, trainSampleIndices, ...
    trainDigitLabels, testInputBatch, testTargetBatch, ...
    testSampleIndices, testDigitLabels, digitLabels, letterNames, ...
    mapData.HRS2LRS, resultsDir, config);
end

function validateImageMatrix(images, variableName)
if ~ismatrix(images) || size(images, 1) < 2 || ...
        ~all(abs(images(1, :) - 1) < 1e-8)
    error('%s must have a leading bias row followed by image pixels.', variableName);
end
pixelCount = size(images, 1) - 1;
imageSide = sqrt(pixelCount);
if imageSide ~= floor(imageSide)
    error('%s pixels must form a square image.', variableName);
end
end

function targetPath = findLetterTarget(outputsDir, letter, outputSize)
targetCandidates = { ...
    fullfile(outputsDir, sprintf('%s_%d.png', letter, outputSize)), ...
    fullfile(outputsDir, sprintf('%s.png', letter)), ...
    fullfile(outputsDir, sprintf('%s_8.png', letter))};
for candidateIndex = 1:numel(targetCandidates)
    if isfile(targetCandidates{candidateIndex})
        targetPath = targetCandidates{candidateIndex};
        return;
    end
end
error('No target image found for %s at size %d under %s.', ...
    letter, outputSize, outputsDir);
end

function [inputs, targets, sampleIndices, sampleDigitLabels] = makeBatch( ...
    images, labels, samplesPerDigit, digitLabels, targetByDigit, inputSize)
sampleIndicesByDigit = cell(numel(digitLabels), 1);
for digitPosition = 1:numel(digitLabels)
    digitIndices = find(labels == digitLabels(digitPosition));
    if isempty(digitIndices)
        error('No examples found for digit %d.', digitLabels(digitPosition));
    end
    sampleIndicesByDigit{digitPosition} = ...
        digitIndices(1:min(numel(digitIndices), samplesPerDigit));
end
sampleIndices = vertcat(sampleIndicesByDigit{:});
sampleDigitLabels = labels(sampleIndices);
inputs = zeros(inputSize^2, numel(sampleIndices));
targets = zeros(size(targetByDigit, 1), numel(sampleIndices));
for sampleIndex = 1:numel(sampleIndices)
    sourceImage = reshape(images(2:end, sampleIndices(sampleIndex)), 24, 24);
    resizedInput = imresize(sourceImage, [inputSize, inputSize], 'bilinear');
    inputs(:, sampleIndex) = resizedInput(:);
    digitPosition = find(digitLabels == sampleDigitLabels(sampleIndex), 1);
    targets(:, sampleIndex) = targetByDigit(:, digitPosition);
end
end

function runHardwareTraining(inputBatch, targetBatch, trainSampleIndices, ...
    trainDigitLabels, testInputBatch, testTargetBatch, testSampleIndices, ...
    testDigitLabels, digitLabels, letterNames, crossbarMap, resultsDir, config)
requiredFunctions = {'deviceWeightMapping', 'connectArduino', 'connectSMU', ...
    'turnOnSMUChannels', 'turnOffSMUChannels', 'setDeviceN', 'resetDeviceN', ...
    'applyVoltage', 'applyReadMeasurement', 'volt'};
for functionIndex = 1:numel(requiredFunctions)
    if exist(requiredFunctions{functionIndex}, 'file') == 0
        error('Required hardware helper is not on the MATLAB path: %s', ...
            requiredFunctions{functionIndex});
    end
end

Nin = size(inputBatch, 1);
Nout = size(targetBatch, 1);
nTrain = size(inputBatch, 2);
if config.inputSize > size(crossbarMap, 1) || ...
        config.outputSize^2 > size(crossbarMap, 2) * size(crossbarMap, 1)
    error('The configured weight matrix does not fit the crossbar dimensions.');
end

assumedWorkingMap = ones(size(crossbarMap));
[deviceMap, ~, ~] = deviceWeightMapping(assumedWorkingMap, Nout, Nin, 1, 1);

fprintf('Connecting to Arduino...\n');
arduinoObj = connectArduino;
arduinoCleanup = onCleanup(@() delete(arduinoObj));
fprintf('Connecting to SMU...\n');
smu = connectSMU;
fopen(smu);
if ~strcmpi(get(smu, 'Status'), 'open')
    error('Could not open the SMU VISA connection.');
end
smuCleanup = onCleanup(@() safeShutdownSMU(smu));
fprintf('SMU communication verified: %s\n', strtrim(query(smu, '*IDN?')));

fwrite(smu, ':OUTP1 OFF; OUTP2 OFF');
fwrite(smu, ':sour1:func:mode volt');
fwrite(smu, ':sour1:volt 0');
fwrite(smu, [':sens1:curr:prot ', num2str(config.iccBias, '%.12g')]);
source1Voltage = str2double(strtrim(query(smu, ':sour1:volt?')));
source1Limit = str2double(strtrim(query(smu, ':sens1:curr:prot?')));
source2Limit = str2double(strtrim(query(smu, ':sens2:curr:prot?')));
if ~isfinite(source1Voltage) || abs(source1Voltage) > 1e-3 || ...
        ~isfinite(source1Limit) || source1Limit > config.iccBias * 1.01 || ...
        ~isfinite(source2Limit) || source2Limit <= 0
    error('SMU did not confirm the requested safe startup settings.');
end
fprintf('Preserving source-2 bias compliance: %.6g A.\n', source2Limit);
turnOnSMUChannels(smu);

fprintf('Initializing %d weights on the 2x2 map...\n', Nout * Nin);
conductance = zeros(Nout, Nin);
for outputIndex = 1:Nout
    for inputIndex = 1:Nin
        conductance(outputIndex, inputIndex) = programOneDevice( ...
            arduinoObj, smu, deviceMap, outputIndex, inputIndex, config.gBias, config);
    end
    fprintf('Initialized %d/%d output rows.\n', outputIndex, Nout);
end
weights = conductanceToWeights(conductance, config.gMin, config.gMax);

if ~isfolder(resultsDir)
    mkdir(resultsDir);
end
metricsFile = fullfile(resultsDir, 'mse_by_epoch.csv');
metricsHandle = fopen(metricsFile, 'w');
if metricsHandle < 0
    error('Could not create metrics file: %s', metricsFile);
end
metricsCleanup = onCleanup(@() fclose(metricsHandle));
fprintf(metricsHandle, ...
    ['epoch,training_pixel_mse,test_pixel_mse,samples_above_threshold,', ...
    'zero_gradient_samples,updates_applied\n']);
updatesFile = fullfile(resultsDir, 'weight_updates.csv');
updatesHandle = fopen(updatesFile, 'w');
if updatesHandle < 0
    error('Could not create update log: %s', updatesFile);
end
updatesCleanup = onCleanup(@() fclose(updatesHandle));
fprintf(updatesHandle, ['epoch,train_sample_index,digit_label,output_index,', ...
    'input_index,sample_mse,gradient,desired_weight,measured_conductance_S,', ...
    'measured_weight,read_attempts,read_voltage_V,read_current_A,', ...
    'resistance_ohm\n']);

updateCount = 0;
trainingMSE = zeros(config.epochs, 1);
testMSE = zeros(config.epochs, 1);
for epoch = 1:config.epochs
    sampleOrder = randperm(nTrain);
    samplesAboveThreshold = 0;
    zeroGradientCount = 0;
    epochUpdateCount = 0;
    for samplePosition = 1:nTrain
        sampleIndex = sampleOrder(samplePosition);
        inputVector = inputBatch(:, sampleIndex);
        targetVector = targetBatch(:, sampleIndex);
        residual = weights * inputVector - targetVector;
        sampleMSE = mean(residual.^2);
        if sampleMSE <= config.sampleMSEThreshold
            continue;
        end
        samplesAboveThreshold = samplesAboveThreshold + 1;

        outputIndex = randi(Nout);
        inputIndex = randi(Nin);
        gradient = residual(outputIndex) * inputVector(inputIndex);
        if gradient == 0
            zeroGradientCount = zeroGradientCount + 1;
            continue;
        end

        oldWeight = weights(outputIndex, inputIndex);
        desiredWeight = min(max(oldWeight - config.learningRate * gradient, ...
            -config.weightLimit), config.weightLimit);
        desiredConductance = config.gBias + desiredWeight * config.gHalfRange;
        [measuredConductance, readVoltage, readCurrent, resistance, readAttempts] = ...
            programOneDevice( ...
            arduinoObj, smu, deviceMap, outputIndex, inputIndex, ...
            desiredConductance, config);
        conductance(outputIndex, inputIndex) = measuredConductance;
        weights(outputIndex, inputIndex) = conductanceToWeights( ...
            measuredConductance, config.gMin, config.gMax);
        updateCount = updateCount + 1;
        epochUpdateCount = epochUpdateCount + 1;
        fprintf(updatesHandle, ...
            '%d,%d,%d,%d,%d,%.12g,%.12g,%.12g,%.12g,%.12g,%d,%.12g,%.12g,%.12g\n', ...
            epoch, trainSampleIndices(sampleIndex), trainDigitLabels(sampleIndex), ...
            outputIndex, inputIndex, sampleMSE, gradient, desiredWeight, ...
            measuredConductance, weights(outputIndex, inputIndex), readAttempts, ...
            readVoltage, readCurrent, resistance);
    end

    trainingMSE(epoch) = mean((weights * inputBatch - targetBatch).^2, 'all');
    testMSE(epoch) = mean((weights * testInputBatch - testTargetBatch).^2, 'all');
    fprintf(metricsHandle, '%d,%.12g,%.12g,%d,%d,%d\n', epoch, ...
        trainingMSE(epoch), testMSE(epoch), samplesAboveThreshold, ...
        zeroGradientCount, epochUpdateCount);
    fprintf(['Epoch %d/%d | train MSE %.8g | test MSE %.8g | ', ...
        'above threshold %d/%d | zero gradients %d | updates %d\n'], ...
        epoch, config.epochs, trainingMSE(epoch), testMSE(epoch), ...
        samplesAboveThreshold, nTrain, zeroGradientCount, epochUpdateCount);
end

save(fullfile(resultsDir, 'final_state.mat'), 'weights', 'conductance', ...
    'trainingMSE', 'testMSE', 'inputBatch', 'targetBatch', ...
    'trainSampleIndices', 'trainDigitLabels', 'testInputBatch', ...
    'testTargetBatch', 'testSampleIndices', 'testDigitLabels', 'config');
testPredictions = weights * testInputBatch;
for digitPosition = 1:numel(digitLabels)
    testRows = find(testDigitLabels == digitLabels(digitPosition));
    for sampleNumber = 1:min(3, numel(testRows))
        row = testRows(sampleNumber);
        image = reshape(min(max(testPredictions(:, row), 0), 1), ...
            config.outputSize, config.outputSize);
        imagePath = fullfile(resultsDir, sprintf( ...
            'test_digit%d_sample%02d_to_%s.png', digitLabels(digitPosition), ...
            sampleNumber, upper(letterNames{digitPosition})));
        imwrite(image, imagePath);
    end
end
figure;
plot(1:config.epochs, trainingMSE, '-', 1:config.epochs, testMSE, '--');
xlabel('Epoch');
ylabel('Pixel MSE');
legend('Training', 'Test');
grid on;
saveas(gcf, fullfile(resultsDir, 'training_and_test_mse.png'));
clear metricsCleanup updatesCleanup smuCleanup arduinoCleanup;
fprintf('Completed %d nonzero-gradient hardware updates. Results: %s\n', ...
    updateCount, resultsDir);
end

function weights = conductanceToWeights(conductance, gMin, gMax)
weights = 2 * (conductance - gMin) / (gMax - gMin) - 1;
weights = min(max(weights, -1), 1);
end

function [measuredConductance, readVoltage, readCurrent, resistance, readAttempts] = programOneDevice( ...
    arduinoObj, smu, deviceMap, outputIndex, inputIndex, targetConductance, config)
row = deviceMap(1, outputIndex, inputIndex);
column = deviceMap(2, outputIndex, inputIndex);
targetConductance = min(max(targetConductance, config.gMin), config.gMax);
targetVoltage = volt(targetConductance / config.gVoltageScale);

setDeviceN(arduinoObj, row, column);
applyVoltage(smu, config.voltageToLRS, config.timestep, config.iccSet);
resetDeviceN(arduinoObj, row, column);
applyVoltage(smu, targetVoltage, config.timestep, config.iccReset);
setDeviceN(arduinoObj, row, column);
pause(config.postPulseReadDelay);
voltageTolerance = max(config.readVoltageAbsoluteTolerance, ...
    config.readVoltageRelativeTolerance * abs(config.voltageRead));
readVoltage = NaN;
readCurrent = NaN;
resistance = NaN;
readAttempts = 0;
readbackValid = false;
lastReadFailure = 'no read was completed';
for readAttempt = 1:config.readbackMaxAttempts
    fwrite(smu, ':trig:coun 1');
    fwrite(smu, ':trig:sour tim');
    fwrite(smu, sprintf(':trig:tim %10.6e', config.timestep));
    [readVoltage, readCurrent, resistance, ~] = applyReadMeasurement( ...
        smu, config.voltageRead, config.timestep, config.iccRead);
    readAttempts = readAttempt;

    if ~isscalar(resistance) || ~isfinite(resistance) || resistance <= 0 || ...
            ~isscalar(readVoltage) || ~isfinite(readVoltage) || ...
            ~isscalar(readCurrent) || ~isfinite(readCurrent)
        error('Invalid readback at (%d,%d), attempt %d: R=%s, V=%s, I=%s.', ...
            row, column, readAttempt, mat2str(resistance), ...
            mat2str(readVoltage), mat2str(readCurrent));
    end

    if abs(readVoltage - config.voltageRead) > voltageTolerance
        lastReadFailure = sprintf('voltage droop (requested %.6g V)', ...
            config.voltageRead);
    elseif abs(readCurrent) >= ...
            config.readCurrentComplianceFraction * config.iccRead
        lastReadFailure = sprintf('current near compliance limit %.6g A', ...
            config.iccRead);
    else
        readbackValid = true;
        break;
    end

    if readAttempt < config.readbackMaxAttempts
        fprintf(['Readback at (%d,%d) failed on attempt %d/%d: ', ...
            '%s; measured V=%.6g V, I=%.6g A, R=%.6g Ohm. ', ...
            'Waiting %.3g s, then retrying read only.\n'], ...
            row, column, readAttempt, config.readbackMaxAttempts, ...
            lastReadFailure, readVoltage, readCurrent, resistance, ...
            config.readbackRetryDelay);
        pause(config.readbackRetryDelay);
    end
end
if ~readbackValid
    error(['Readback at (%d,%d) remained invalid after %d read-only attempts: ', ...
        '%s; last measured V=%.6g V, I=%.6g A, R=%.6g Ohm. ', ...
        'No programming pulse was repeated.'], ...
        row, column, readAttempts, lastReadFailure, ...
        readVoltage, readCurrent, resistance);
end
measuredConductance = 1 / resistance;
if ~isfinite(measuredConductance) || measuredConductance < config.gMin || ...
        measuredConductance > config.gMax
    error('Measured conductance %.6g uS at (%d,%d) is outside the calibrated range.', ...
        measuredConductance * 1e6, row, column);
end
end

function safeShutdownSMU(smu)
try
    if strcmpi(get(smu, 'Status'), 'open')
        turnOffSMUChannels(smu);
    end
catch shutdownError
    warning('%s', ['Could not turn off SMU outputs cleanly: ', shutdownError.message]);
end
try
    if strcmpi(get(smu, 'Status'), 'open')
        fclose(smu);
    end
catch closeError
    warning('%s', ['Could not close the SMU connection cleanly: ', closeError.message]);
end
end