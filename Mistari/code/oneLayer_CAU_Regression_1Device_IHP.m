%% One-device-per-weight hardware regression: MNIST 1/2/3 -> C/A/U
% This adapts the IHP hardware coordinate-update flow to linear pixel
% regression. Each weight uses one physical device. Its signed software
% weight is conductance relative to a fixed midpoint:
%   W = (G - G_BIAS) / G_HALF_RANGE
% Review the device calibration, then set CAU_RUN_HARDWARE=1 before programming.
%setenv('CAU_RUN_HARDWARE', '0');
%run('C:\Users\tbe\CAU_task\Mistari\code\oneLayer_CAU_Regression_1Device_IHP.m');

clearvars;
clc;
rng(123);
fprintf('Running script: %s\n', mfilename('fullpath'));

%% Paths and experiment configuration
scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));
funcDir = fullfile(scriptDir, 'func');
ihpFunctionsDir = fullfile(fileparts(scriptDir), 'IHPfunctions');
if isfolder(scriptDir)
    addpath(scriptDir, '-begin');
end
if ~isfolder(ihpFunctionsDir)
    ihpFunctionsDir = fullfile(scriptDir, 'IHPfunctions');
end
if isfolder(funcDir)
    addpath(funcDir);
end
if isfolder(ihpFunctionsDir)
    addpath(ihpFunctionsDir);
end

inputFile = fullfile(funcDir, 'MNISTdataset0123.mat');
if ~isfile(inputFile)
    inputFile = fullfile(projectRoot, 'Inputs', 'MNISTdataset0123.mat');
end
outputsDir = fullfile(scriptDir, 'Outputs');
if ~isfolder(outputsDir) && isfolder(fullfile(projectRoot, 'Outputs'))
    outputsDir = fullfile(projectRoot, 'Outputs');
end
resultsDir = fullfile(outputsDir, 'hardware_regression_1device');

INPUT_SIZE = 8;
OUTPUT_SIZE = 8;
EPOCHS = 500;
LEARNING_RATE = 1.0;
SAMPLE_MSE_THRESHOLD = 0.2; %try 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2
MAX_SAMPLES_PER_DIGIT = []; % [] uses every training example; set e.g. 100 to cap each class
% Enable from MATLAB with setenv('CAU_RUN_HARDWARE', '1') before running.
RUN_HARDWARE = strcmp(getenv('CAU_RUN_HARDWARE'), '1');
fprintf('Sample MSE gate: update samples only when MSE > %.4g.\n', ...
    SAMPLE_MSE_THRESHOLD);

% Keep target conductance inside volt.m's calibrated A2..A1 range.
% These bounds correspond to its 3.72159 uS and 138.826 uS endpoints.
G_VOLTAGE_SCALE = 160e-6;
G_MIN = 3.72159e-6;
G_MAX = 1.38826e-4;
G_BIAS = (G_MIN + G_MAX) / 2;
G_HALF_RANGE = (G_MAX - G_MIN) / 2;
WEIGHT_LIMIT = 1.0;
voltageToLRS = 1.5;
voltageRead = 0.2;
timestep = 1e-1;
IccSet = 1e-4;
IccReset = 2e-4;
IccRead = 5e-5;
POST_PULSE_READ_DELAY = 1.0;
READ_CURRENT_COMPLIANCE_FRACTION = 0.95;
READ_VOLTAGE_RELATIVE_TOLERANCE = 0.05;
READ_VOLTAGE_ABSOLUTE_TOLERANCE = 5e-3;

%% Load training and held-out test examples
if ~isfile(inputFile)
    error('MNIST input file not found: %s', inputFile);
end
matData = load(inputFile, 'imagesTrain', 'labelsTrain', 'imagesTest', 'labelsTest');
if ~isfield(matData, 'imagesTrain') || ~isfield(matData, 'labelsTrain') || ...
        ~isfield(matData, 'imagesTest') || ~isfield(matData, 'labelsTest')
    error('MAT file must contain training and test images and labels.');
end
imagesTrain = double(matData.imagesTrain);
labelsTrain = double(matData.labelsTrain(:));
imagesTest = double(matData.imagesTest);
labelsTest = double(matData.labelsTest(:));
if size(imagesTrain, 1) < 2 || ~all(abs(imagesTrain(1, :) - 1) < 1e-8)
    error('Expected imagesTrain rows to be bias followed by 24x24 pixels.');
end
if size(imagesTest, 1) ~= size(imagesTrain, 1) || ...
        ~all(abs(imagesTest(1, :) - 1) < 1e-8)
    error('Expected imagesTest to use the same bias-plus-pixels layout as imagesTrain.');
end
if ~isempty(MAX_SAMPLES_PER_DIGIT)
    validateattributes(MAX_SAMPLES_PER_DIGIT, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'integer', 'positive'});
end

digitLabels = [1, 2, 3];
letterNames = {'c', 'a', 'u'};
targetByDigit = zeros(OUTPUT_SIZE * OUTPUT_SIZE, numel(digitLabels));
for digitPosition = 1:numel(digitLabels)
    letter = letterNames{digitPosition};
    targetPath = fullfile(outputsDir, ...
        sprintf('%s_%d.png', letter, OUTPUT_SIZE));
    if ~isfile(targetPath)
        error('Letter target not found: %s', targetPath);
    end
    targetImage = imread(targetPath);
    if ndims(targetImage) == 3
        targetImage = rgb2gray(targetImage(:, :, 1:3));
    end
    targetImage = im2double(imresize(targetImage, ...
        [OUTPUT_SIZE, OUTPUT_SIZE], 'bilinear'));
    targetByDigit(:, digitPosition) = targetImage(:);
end

sampleIndicesByDigit = cell(numel(digitLabels), 1);
for digitPosition = 1:numel(digitLabels)
    digitIndices = find(labelsTrain == digitLabels(digitPosition));
    if isempty(digitIndices)
        error('No training examples found for digit %d.', digitLabels(digitPosition));
    end
    if ~isempty(MAX_SAMPLES_PER_DIGIT)
        digitIndices = digitIndices(1:min(numel(digitIndices), MAX_SAMPLES_PER_DIGIT));
    end
    sampleIndicesByDigit{digitPosition} = digitIndices;
end
sampleIndices = sort(vertcat(sampleIndicesByDigit{:}));
sampleDigitLabels = labelsTrain(sampleIndices);
testSampleIndices = find(ismember(labelsTest, digitLabels));
if isempty(testSampleIndices)
    error('No held-out test examples found for digits 1, 2, or 3.');
end
testDigitLabels = labelsTest(testSampleIndices);

inputBatch = zeros(INPUT_SIZE * INPUT_SIZE, numel(sampleIndices));
targetBatch = zeros(OUTPUT_SIZE * OUTPUT_SIZE, numel(sampleIndices));
for sample = 1:numel(sampleIndices)
    sourceImage = reshape(imagesTrain(2:end, sampleIndices(sample)), 24, 24);
    resizedInput = imresize(sourceImage, [INPUT_SIZE, INPUT_SIZE], 'bilinear');
    inputBatch(:, sample) = resizedInput(:);
    digitPosition = find(digitLabels == sampleDigitLabels(sample), 1);
    targetBatch(:, sample) = targetByDigit(:, digitPosition);
end

testInputBatch = zeros(INPUT_SIZE * INPUT_SIZE, numel(testSampleIndices));
testTargetBatch = zeros(OUTPUT_SIZE * OUTPUT_SIZE, numel(testSampleIndices));
for sample = 1:numel(testSampleIndices)
    sourceImage = reshape(imagesTest(2:end, testSampleIndices(sample)), 24, 24);
    resizedInput = imresize(sourceImage, [INPUT_SIZE, INPUT_SIZE], 'bilinear');
    testInputBatch(:, sample) = resizedInput(:);
    digitPosition = find(digitLabels == testDigitLabels(sample), 1);
    testTargetBatch(:, sample) = targetByDigit(:, digitPosition);
end

Nin = size(inputBatch, 1);
Nout = size(targetBatch, 1);
requiredDevices = Nin * Nout;
fprintf('Training examples: %d (digits 1/2/3: %d/%d/%d); held-out test examples: %d.\n', ...
    numel(sampleIndices), sum(sampleDigitLabels == 1), ...
    sum(sampleDigitLabels == 2), sum(sampleDigitLabels == 3), numel(testSampleIndices));
if ~isempty(MAX_SAMPLES_PER_DIGIT)
    fprintf('Training capped at %d examples per digit.\n', MAX_SAMPLES_PER_DIGIT);
end
fprintf('Linear map: %d inputs -> %d output pixels; one device per weight (%d devices).\n', ...
    Nin, Nout, requiredDevices);
fprintf('Signed weight encoding: W = (G - %.1f uS) / %.1f uS.\n', ...
    G_BIAS * 1e6, G_HALF_RANGE * 1e6);

mapPath = fullfile(ihpFunctionsDir, 'crossbarMap.mat');
if ~isfile(mapPath)
    mapPath = fullfile(scriptDir, 'crossbarMap.mat');
end
if ~isfile(mapPath)
    mapPath = fullfile(pwd, 'crossbarMap.mat');
end
if isfile(mapPath)
    mapData = load(mapPath, 'HRS2LRS');
    if ~isfield(mapData, 'HRS2LRS')
        error('crossbarMap.mat must contain HRS2LRS.');
    end
    availableDevices = numel(mapData.HRS2LRS);
    fprintf(['Assuming all %d crossbar sites work (the current map flags %d); ', ...
        'this model needs %d.\n'], ...
        availableDevices, nnz(mapData.HRS2LRS), requiredDevices);
else
    mapData = struct();
    availableDevices = NaN;
    fprintf('crossbarMap.mat not found; map capacity cannot be checked yet.\n');
end

if ~RUN_HARDWARE
    fprintf(['Hardware is disabled. Inspect the configuration and device calibration, ', ...
        'then setenv(''CAU_RUN_HARDWARE'', ''1'') in MATLAB and rerun.\n']);
    return;
end

if ~isfile(mapPath)
    error('crossbarMap.mat not found. Generate the hardware map before running.');
end
if requiredDevices > availableDevices
    error('This model needs %d sites, but the crossbar has only %d.', ...
        requiredDevices, availableDevices);
end

hardwareConfig = struct( ...
    'inputSize', INPUT_SIZE, 'outputSize', OUTPUT_SIZE, 'epochs', EPOCHS, ...
    'maxSamplesPerDigit', MAX_SAMPLES_PER_DIGIT, ...
    'learningRate', LEARNING_RATE, 'sampleMSEThreshold', SAMPLE_MSE_THRESHOLD, ...
    'gMin', G_MIN, 'gMax', G_MAX, ...
    'gVoltageScale', G_VOLTAGE_SCALE, 'gBias', G_BIAS, ...
    'gHalfRange', G_HALF_RANGE, 'weightLimit', WEIGHT_LIMIT, ...
    'voltageToLRS', voltageToLRS, 'voltageRead', voltageRead, ...
    'timestep', timestep, 'iccSet', IccSet, 'iccReset', IccReset, ...
    'iccRead', IccRead, 'postPulseReadDelay', POST_PULSE_READ_DELAY, ...
    'iccBias', min(IccSet, IccReset), ...
    'readCurrentComplianceFraction', READ_CURRENT_COMPLIANCE_FRACTION, ...
    'readVoltageRelativeTolerance', READ_VOLTAGE_RELATIVE_TOLERANCE, ...
    'readVoltageAbsoluteTolerance', READ_VOLTAGE_ABSOLUTE_TOLERANCE);
runHardwareTraining(inputBatch, targetBatch, sampleIndices, sampleDigitLabels, ...
    testInputBatch, testTargetBatch, testSampleIndices, testDigitLabels, ...
    digitLabels, letterNames, mapData.HRS2LRS, resultsDir, hardwareConfig);
return;

function runHardwareTraining(inputBatch, targetBatch, sampleIndices, sampleDigitLabels, ...
    testInputBatch, testTargetBatch, testSampleIndices, testDigitLabels, ...
    digitLabels, letterNames, crossbarMap, resultsDir, config)
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
if config.inputSize > size(crossbarMap, 1) || config.outputSize^2 > size(crossbarMap, 2) * size(crossbarMap, 1)
    error('The configured weight matrix does not fit the crossbar dimensions.');
end

% Deliberately ignore the current usable-device mask for this idealized run.
assumedWorkingMap = ones(size(crossbarMap));
[deviceMap, ~, ~] = deviceWeightMapping(assumedWorkingMap, Nout, Nin, 1, 1);

fprintf('Hardware mode enabled. Beginning instrument setup.\n');
drawnow;
fprintf('Connecting to Arduino...\n');
drawnow;
arduinoObj = connectArduino;
arduinoCleanup = onCleanup(@() delete(arduinoObj));
fprintf('Opening SMU connection...\n');
drawnow;
fprintf('connectSMU resolved to: %s\n', which('connectSMU'));
smu = connectSMU;
fprintf('Opening VISA session (status before fopen: %s)...\n', get(smu, 'Status'));
fopen(smu);
if ~strcmpi(get(smu, 'Status'), 'open')
    error('Could not open the SMU VISA connection.');
end
smuCleanup = onCleanup(@() safeShutdownSMU(smu));
smuIdentity = strtrim(query(smu, '*IDN?'));
fprintf('SMU communication verified: %s\n', smuIdentity);

% Start from outputs off; configure and verify limits before enabling them.
fprintf('Configuring safe startup voltage and current limits...\n');
drawnow;
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
fprintf('Preserving existing source-2 bias compliance: %.6g A.\n', source2Limit);
turnOnSMUChannels(smu);

fprintf('Initializing %d devices to the zero-weight conductance midpoint...\n', ...
    Nout * Nin);
drawnow;
conductance = zeros(Nout, Nin);
for outputIndex = 1:Nout
    for inputIndex = 1:Nin
        conductance(outputIndex, inputIndex) = programOneDevice( ...
            arduinoObj, smu, deviceMap, outputIndex, inputIndex, config.gBias, config);
    end
    fprintf('Initialization progress: %d/%d output rows (%d/%d devices).\n', ...
        outputIndex, Nout, outputIndex * Nin, Nout * Nin);
    drawnow;
end
weights = conductanceToWeights(conductance, config.gMin, config.gMax);
initialPredictions = weights * inputBatch;
initialTestPredictions = weights * testInputBatch;
initialTrainingMSE = mean((initialPredictions - targetBatch).^2, 'all');
initialTestMSE = mean((initialTestPredictions - testTargetBatch).^2, 'all');
fprintf('Initial training pixel MSE: %.8g; test pixel MSE: %.8g.\n', ...
    initialTrainingMSE, initialTestMSE);
fprintf('Threshold %.6g: %d/%d training samples currently trigger updates.\n', ...
    config.sampleMSEThreshold, ...
    sum(mean((initialPredictions - targetBatch).^2, 1) > config.sampleMSEThreshold), ...
    nTrain);
drawnow;

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
fprintf(updatesHandle, ['epoch,training_sample_index,digit_label,output_index,', ...
    'input_index,sample_mse,gradient,desired_weight,measured_conductance_S,', ...
    'measured_weight\n']);

updateCount = 0;
mseHistory = zeros(config.epochs, 1);
testMSEHistory = zeros(config.epochs, 1);
fprintf('Starting training for %d epochs.\n', config.epochs);
drawnow;
for epoch = 1:config.epochs
    sampleOrder = randperm(nTrain);
    samplesAboveThreshold = 0;
    zeroGradientCount = 0;
    epochUpdateCount = 0;
    for samplePosition = 1:numel(sampleOrder)
        sampleIndex = sampleOrder(samplePosition);
        inputVector = inputBatch(:, sampleIndex);
        targetVector = targetBatch(:, sampleIndex);
        prediction = weights * inputVector;
        residual = prediction - targetVector;
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
        measuredConductance = programOneDevice( ...
            arduinoObj, smu, deviceMap, outputIndex, inputIndex, desiredConductance, config);
        conductance(outputIndex, inputIndex) = measuredConductance;
        weights(outputIndex, inputIndex) = conductanceToWeights( ...
            measuredConductance, config.gMin, config.gMax);

        updateCount = updateCount + 1;
        epochUpdateCount = epochUpdateCount + 1;
        fprintf(updatesHandle, ...
            '%d,%d,%d,%d,%d,%.12g,%.12g,%.12g,%.12g,%.12g\n', ...
            epoch, sampleIndices(sampleIndex), sampleDigitLabels(sampleIndex), ...
            outputIndex, inputIndex, sampleMSE, gradient, desiredWeight, ...
            measuredConductance, weights(outputIndex, inputIndex));
    end

    predictions = weights * inputBatch;
    mseHistory(epoch) = mean((predictions - targetBatch).^2, 'all');
    testPredictions = weights * testInputBatch;
    testMSEHistory(epoch) = mean((testPredictions - testTargetBatch).^2, 'all');
    fprintf(metricsHandle, '%d,%.12g,%.12g,%d,%d,%d\n', ...
        epoch, mseHistory(epoch), testMSEHistory(epoch), samplesAboveThreshold, ...
        zeroGradientCount, epochUpdateCount);
    fprintf(['Epoch %d/%d | training MSE = %.8g | test MSE = %.8g | ', ...
        'above threshold = %d/%d | zero gradients = %d | updates = %d\n'], ...
        epoch, config.epochs, mseHistory(epoch), testMSEHistory(epoch), ...
        samplesAboveThreshold, nTrain, zeroGradientCount, epochUpdateCount);
end

save(fullfile(resultsDir, 'final_state.mat'), 'weights', 'conductance', ...
    'mseHistory', 'testMSEHistory', 'inputBatch', 'targetBatch', ...
    'sampleIndices', 'sampleDigitLabels', 'testInputBatch', 'testTargetBatch', ...
    'testSampleIndices', 'testDigitLabels', 'digitLabels', 'letterNames', 'config');

testPredictions = weights * testInputBatch;
for digitPosition = 1:numel(digitLabels)
    testRows = find(testDigitLabels == digitLabels(digitPosition));
    for sampleNumber = 1:min(3, numel(testRows))
        testRow = testRows(sampleNumber);
        reconstruction = reshape(min(max(testPredictions(:, testRow), 0), 1), ...
            config.outputSize, config.outputSize);
        imagePath = fullfile(resultsDir, sprintf( ...
            'test_digit%d_sample%02d_to_%s.png', digitLabels(digitPosition), ...
            sampleNumber, upper(letterNames{digitPosition})));
        imwrite(reconstruction, imagePath);
    end
end

figure;
plot(1:config.epochs, mseHistory, '-', 1:config.epochs, testMSEHistory, '--');
xlabel('Epoch');
ylabel('Pixel MSE');
legend('Training', 'Test');
title('MNIST digit to letter regression: one device per weight');
grid on;
saveas(gcf, fullfile(resultsDir, 'training_and_test_mse.png'));
clear metricsCleanup;
clear updatesCleanup;
clear smuCleanup;
clear arduinoCleanup;
fprintf('Total nonzero-gradient hardware updates: %d.\n', updateCount);
fprintf('Hardware regression finished. Results: %s\n', resultsDir);
end

%% Local helpers
function weights = conductanceToWeights(conductance, gMin, gMax)
weights = 2 * (conductance - gMin) / (gMax - gMin) - 1;
weights = min(max(weights, -1), 1);
end

function measuredConductance = programOneDevice( ...
    arduinoObj, smu, deviceMap, outputIndex, inputIndex, targetConductance, config)
row = deviceMap(1, outputIndex, inputIndex);
column = deviceMap(2, outputIndex, inputIndex);
targetConductance = min(max(targetConductance, config.gMin), config.gMax);
targetVoltage = volt(targetConductance / config.gVoltageScale);

% Re-establish a known LRS, then RESET toward the calibrated target state.
setDeviceN(arduinoObj, row, column);
applyVoltage(smu, config.voltageToLRS, config.timestep, config.iccSet);
resetDeviceN(arduinoObj, row, column);
applyVoltage(smu, targetVoltage, config.timestep, config.iccReset);
setDeviceN(arduinoObj, row, column);
pause(config.postPulseReadDelay);
[readVoltage, readCurrent, resistance, ~] = applyReadMeasurement( ...
    smu, config.voltageRead, config.timestep, config.iccRead);
if ~isscalar(resistance) || ~isfinite(resistance) || resistance <= 0 || ...
    ~isscalar(readVoltage) || ~isfinite(readVoltage) || ...
    ~isscalar(readCurrent) || ~isfinite(readCurrent)
    error(['Invalid resistance read at device (%d, %d): R=%s Ohm, ', ...
        'V=%s V, I=%s A; target G=%.6g uS, reset pulse=%.4g V, ', ...
        'current limits SET/RESET/READ=%.4g/%.4g/%.4g A. Aborting without retry.'], ...
        row, column, mat2str(resistance), mat2str(readVoltage), ...
        mat2str(readCurrent), targetConductance * 1e6, targetVoltage, ...
        config.iccSet, config.iccReset, config.iccRead);
end
voltageTolerance = max(config.readVoltageAbsoluteTolerance, ...
    config.readVoltageRelativeTolerance * abs(config.voltageRead));
if abs(readVoltage - config.voltageRead) > voltageTolerance
    error(['Read voltage droop at device (%d, %d): requested %.6g V, ', ...
        'measured %.6g V (tolerance %.6g V); current %.6g A. ', ...
        'Treat readback as invalid; no retry.'], ...
        row, column, config.voltageRead, readVoltage, voltageTolerance, readCurrent);
end
if abs(readCurrent) >= config.readCurrentComplianceFraction * config.iccRead
    error(['Read current near compliance at device (%d, %d): %.6g A of read limit %.6g A. ', ...
        'Treat resistance as current-limited and invalid; no retry.'], ...
        row, column, readCurrent, config.iccRead);
end
measuredConductance = 1 / resistance;
if ~isfinite(measuredConductance) || measuredConductance < config.gMin || ...
        measuredConductance > config.gMax
    error(['Measured conductance %.6g uS at device (%d, %d) is outside ', ...
        'the calibrated range [%.6g, %.6g] uS; no clamping or retry.'], ...
        measuredConductance * 1e6, row, column, config.gMin * 1e6, config.gMax * 1e6);
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
