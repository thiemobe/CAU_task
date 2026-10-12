%% Software-only test for thresholded MNIST 1/2/3 -> C/A/U regression
% This mirrors the hardware script's sample-MSE gate and random-coordinate
% gradient update, but uses ideal software weights and never accesses devices.

clearvars;
clc;
rng(123);
fprintf('Running software-only test: %s\n', mfilename('fullpath'));

%% Paths and test configuration
scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDir));
funcDir = fullfile(scriptDir, 'func');
if isfolder(funcDir)
    addpath(funcDir);
end

inputFile = fullfile(funcDir, 'MNISTdataset0123.mat');
if ~isfile(inputFile)
    inputFile = fullfile(projectRoot, 'Inputs', 'MNISTdataset0123.mat');
end
outputsDir = fullfile(scriptDir, 'Outputs');
if ~isfolder(outputsDir) && isfolder(fullfile(projectRoot, 'Outputs'))
    outputsDir = fullfile(projectRoot, 'Outputs');
end
resultsDir = fullfile(outputsDir, 'software_regression_no_init');

INPUT_SIZE = 8;
OUTPUT_SIZE = 8;
EPOCHS = 500;
LEARNING_RATE = 1.0;
SAMPLE_MSE_THRESHOLD = 0.2;
WEIGHT_LIMIT = 1.0;

%% Load the first training example for each digit and matching letter target
if ~isfile(inputFile)
    error('MNIST input file not found: %s', inputFile);
end
matData = load(inputFile, 'imagesTrain', 'labelsTrain');
if ~isfield(matData, 'imagesTrain') || ~isfield(matData, 'labelsTrain')
    error('MAT file must contain imagesTrain and labelsTrain.');
end
imagesTrain = double(matData.imagesTrain);
labelsTrain = double(matData.labelsTrain(:));
if size(imagesTrain, 1) < 2 || ~all(abs(imagesTrain(1, :) - 1) < 1e-8)
    error('Expected imagesTrain rows to be bias followed by 24x24 pixels.');
end

digitLabels = [1, 2, 3];
letterNames = {'c', 'a', 'u'};
sampleIndices = zeros(1, 3);
inputBatch = zeros(INPUT_SIZE * INPUT_SIZE, 3);
targetBatch = zeros(OUTPUT_SIZE * OUTPUT_SIZE, 3);
for sample = 1:3
    sourceIndex = find(labelsTrain == digitLabels(sample), 1, 'first');
    if isempty(sourceIndex)
        error('No training example found for digit %d.', digitLabels(sample));
    end
    sampleIndices(sample) = sourceIndex - 1;
    sourceImage = reshape(imagesTrain(2:end, sourceIndex), 24, 24);
    resizedInput = imresize(sourceImage, [INPUT_SIZE, INPUT_SIZE], 'bilinear');
    inputBatch(:, sample) = resizedInput(:);

    targetPath = fullfile(outputsDir, ...
        sprintf('%s_%d.png', letterNames{sample}, OUTPUT_SIZE));
    if ~isfile(targetPath)
        error('Letter target not found: %s', targetPath);
    end
    targetImage = imread(targetPath);
    if ndims(targetImage) == 3
        targetImage = rgb2gray(targetImage(:, :, 1:3));
    end
    targetImage = im2double(imresize(targetImage, ...
        [OUTPUT_SIZE, OUTPUT_SIZE], 'bilinear'));
    targetBatch(:, sample) = targetImage(:);
end

fprintf('Software-only mode: no SMU, Arduino, or device initialization.\n');
fprintf('MSE threshold = %.4g; updates occur when sample MSE exceeds it.\n', ...
    SAMPLE_MSE_THRESHOLD);

%% Ideal zero-weight starting point; the hardware script instead measures devices
Nin = size(inputBatch, 1);
Nout = size(targetBatch, 1);
weights = zeros(Nout, Nin);
initialPredictions = weights * inputBatch;
initialSampleMSE = mean((initialPredictions - targetBatch).^2, 1);
fprintf('Initial sample MSEs: digit 1=%.6g, digit 2=%.6g, digit 3=%.6g.\n', ...
    initialSampleMSE(1), initialSampleMSE(2), initialSampleMSE(3));

if ~isfolder(resultsDir)
    mkdir(resultsDir);
end
metricsPath = fullfile(resultsDir, 'training_mse.csv');
metricsHandle = fopen(metricsPath, 'w');
if metricsHandle < 0
    error('Could not create metrics file: %s', metricsPath);
end
metricsCleanup = onCleanup(@() fclose(metricsHandle));
fprintf(metricsHandle, ...
    'epoch,training_pixel_mse,samples_above_threshold,updates_applied\n');

updates = zeros(EPOCHS * size(inputBatch, 2), 7);
updateCount = 0;
mseHistory = zeros(EPOCHS, 1);
for epoch = 1:EPOCHS
    sampleOrder = randperm(size(inputBatch, 2));
    samplesAboveThreshold = 0;
    epochUpdateCount = 0;
    for samplePosition = 1:numel(sampleOrder)
        sampleIndex = sampleOrder(samplePosition);
        inputVector = inputBatch(:, sampleIndex);
        targetVector = targetBatch(:, sampleIndex);
        residual = weights * inputVector - targetVector;
        sampleMSE = mean(residual.^2);
        if sampleMSE <= SAMPLE_MSE_THRESHOLD
            continue;
        end
        samplesAboveThreshold = samplesAboveThreshold + 1;

        activeOutputs = find(abs(residual) > eps);
        activeInputs = find(abs(inputVector) > eps);
        if isempty(activeOutputs) || isempty(activeInputs)
            continue;
        end
        outputIndex = activeOutputs(randi(numel(activeOutputs)));
        inputIndex = activeInputs(randi(numel(activeInputs)));
        gradient = residual(outputIndex) * inputVector(inputIndex);
        weights(outputIndex, inputIndex) = min(max( ...
            weights(outputIndex, inputIndex) - LEARNING_RATE * gradient, ...
            -WEIGHT_LIMIT), WEIGHT_LIMIT);

        updateCount = updateCount + 1;
        epochUpdateCount = epochUpdateCount + 1;
        updates(updateCount, :) = [epoch, sampleIndex, outputIndex, ...
            inputIndex, gradient, sampleMSE, weights(outputIndex, inputIndex)];
    end

    predictions = weights * inputBatch;
    mseHistory(epoch) = mean((predictions - targetBatch).^2, 'all');
    fprintf(metricsHandle, '%d,%.12g,%d,%d\n', epoch, mseHistory(epoch), ...
        samplesAboveThreshold, epochUpdateCount);
    fprintf(['Epoch %d/%d | training pixel MSE = %.8g | ', ...
        'samples above threshold = %d/3 | updates = %d\n'], ...
        epoch, EPOCHS, mseHistory(epoch), samplesAboveThreshold, epochUpdateCount);
end

clear metricsCleanup;
updates = updates(1:updateCount, :);
save(fullfile(resultsDir, 'software_test_state.mat'), 'weights', ...
    'mseHistory', 'updates', 'inputBatch', 'targetBatch', 'sampleIndices', ...
    'digitLabels', 'letterNames', 'INPUT_SIZE', 'OUTPUT_SIZE', ...
    'EPOCHS', 'LEARNING_RATE', 'SAMPLE_MSE_THRESHOLD');

predictions = weights * inputBatch;
for sample = 1:3
    reconstruction = reshape(min(max(predictions(:, sample), 0), 1), ...
        OUTPUT_SIZE, OUTPUT_SIZE);
    imagePath = fullfile(resultsDir, sprintf('digit%d_to_%s.png', ...
        digitLabels(sample), upper(letterNames{sample})));
    imwrite(reconstruction, imagePath);
end

fprintf('Completed %d software coordinate updates. Results: %s\n', ...
    updateCount, resultsDir);