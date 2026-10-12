% Deterministic runner for one set pulse followed by reset pulses
% ----------------------------------------------------------
% - Loops automatically over a list of devices (row/col + repetitions + save name)
% - Applies one 1.5 V SET pulse per device
% - Reads reset pulses from reset_pulse.csv
% - Conductance measured before first experiment (U_read=0.2 V)
% - Conductance_after_10s becomes conductance_before of next experiment
% - Read pulses applied at 1,2,...,10 s after every set/reset pulse
% - Automatic intermediate saving
% ----------------------------------------------------------

%% --- User configuration ---

addpath('\\embserver\emd-public\own-software\IHPboard\IHPfunctions');
info = instrhwinfo('visa')

plotIV = true;
saveFile = true;

read_times_s = 1:10;   % read times (s) after each pulse

% --- Compliance levels ---
I_set   = 1e-4;
I_reset = 1.5 * I_set;
I_read  = 5e-5;
U_read  = 0.2;
read_aper = 0.1;     % read pulse duration / aperture [s]

%% --- Device list ---
% high LRS/HRS ratio row and col_nums: (60, 15), (32, 22), (44,
% 41), (49, 56), (28, 15)
% low LRS/HRS ratio row and col_nums: (50, 12), (20, 21), (57, 27), (45, 40), (1, 46)
devices = struct( ...
    'row',      {60, 32, 44, 49, 28, 50, 20, 57, 45, 1}, ...
    'col',      {15, 22, 41, 56, 15, 12, 21, 27, 40, 46}, ...
    'N_reps',   {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, ...
    'saveName', {'train_samples_dev_60_15_0_6_reset_samples_high_ratio_test.csv', ...
                 'train_samples_dev_32_22_0_6_reset_samples_high_ratio_test.csv', ...
                 'train_samples_dev_44_41_0_6_reset_samples_high_ratio_test.csv', ...
                 'train_samples_dev_49_56_0_6_reset_samples_high_ratio_test.csv', ...
                 'train_samples_dev_28_15_0_6_reset_samples_high_ratio_test.csv', ...
                 'train_samples_dev_50_12_0_6_reset_samples_low_ratio_test.csv', ...
                 'train_samples_dev_20_21_0_6_reset_samples_low_ratio_test.csv', ...
                 'train_samples_dev_57_27_0_6_reset_samples_low_ratio_test.csv', ...
                 'train_samples_dev_45_40_0_6_reset_samples_low_ratio_test.csv', ...
                 'train_samples_dev_1_46_0_6_reset_samples_low_ratio_test.csv'});

%% --- Load pulse design ---
setFilePath = '\\embserver\emd-public\own-software\IHPboard\Thiemo\set_1_5_voltages.csv';
resetFilePath = '\\embserver\emd-public\own-software\IHPboard\Thiemo\reset_0_6_voltages.csv';

if ~isfile(setFilePath)
    error('Missing SET pulse design file: %s', setFilePath);
end
if ~isfile(resetFilePath)
    error('Missing reset pulse file: %s', resetFilePath);
end

set_design = readmatrix(setFilePath); % [V, duration_ms]
reset_design = readmatrix(resetFilePath); % [V, duration_ms]

if size(set_design, 2) < 2 || size(reset_design, 2) < 2
    error('SET and reset pulse files must contain [Voltage, Duration_ms] columns');
end

set_voltage = 1.5;
set_duration_ms = set_design(1, 2);
reset_voltages = reset_design(:, 1);
reset_durations = reset_design(:, 2);

N_reset = length(reset_voltages);

saveDir = 'data';
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

% --- Hardware connection (shared across all devices) ---
if ~exist('arduinoObj', 'var')
    arduinoObj = connectArduino;
end
if ~exist('smu', 'var')
    smu = connectSMU;
end
fopen(smu)
turnOnSMUChannels(smu);

%% --- Loop over devices, one after another ---
for d = 1:numel(devices)
    row_num  = devices(d).row;
    col_num  = devices(d).col;
    savepath = fullfile(saveDir, devices(d).saveName);

    fprintf('\n=== Starting device %d/%d: row=%d, col=%d ===\n', d, numel(devices), row_num, col_num);

    %% --- Build deterministic experiment order ---
    % Apply one SET pulse first, followed by every reset pulse once.
    sequence_voltage = zeros(1 + N_reset, 1);
    sequence_duration = zeros(size(sequence_voltage));
    sequence_mode = strings(size(sequence_voltage));
    sequence_list_index = zeros(size(sequence_voltage));
    sequence_rep_index = zeros(size(sequence_voltage));

    sequence_voltage(1) = set_voltage;
    sequence_duration(1) = set_duration_ms;
    sequence_mode(1) = "set";
    sequence_list_index(1) = 1;
    sequence_rep_index(1) = 1;

    sequence_voltage(2:end) = reset_voltages;
    sequence_duration(2:end) = reset_durations;
    sequence_mode(2:end) = "reset";
    sequence_list_index(2:end) = (1:N_reset)';
    sequence_rep_index(2:end) = 1;

    ExpTable = table(sequence_voltage, sequence_duration, sequence_mode, ...
        sequence_list_index, sequence_rep_index, ...
        'VariableNames', {'Voltage','Duration','Mode','ListIndex','RepIndex'});
    N_total  = height(ExpTable);

    %% --- Prepare storage table ---
    TBL = table();
    TBL.State = strings(N_total,1);
    TBL.ListIndex = ExpTable.ListIndex;
    TBL.RepIndex = ExpTable.RepIndex;
    TBL.V_applied = zeros(N_total,1);
    TBL.PulseDur_s = zeros(N_total,1);

    TBL.U_out_immediate = zeros(N_total,1);
    TBL.I_immediate = zeros(N_total,1);
    TBL.R_immediate = zeros(N_total,1);
    TBL.T_immediate = zeros(N_total,1);

    TBL.R_before = zeros(N_total,1);
    TBL.G_before = zeros(N_total,1);

    % pre-allocate one set of read columns per read time (1..10 s)
    for t = read_times_s
        TBL.(sprintf('U_after_%ds', t)) = zeros(N_total,1);
        TBL.(sprintf('I_after_%ds', t)) = zeros(N_total,1);
        TBL.(sprintf('R_after_%ds', t)) = zeros(N_total,1);
        TBL.(sprintf('G_after_%ds', t)) = zeros(N_total,1);
        TBL.(sprintf('T_after_%ds', t)) = zeros(N_total,1);
    end

    TBL.time_stamp = strings(N_total,1);

    %% --- Initial conductance measurement before first experiment ---
    fprintf("Initial read before first experiment...\n");
    setDeviceN(arduinoObj, row_num, col_num); % Ensure proper addressing
    pause(0.1); % small delay for stability
    [~, ~, R_r0, ~] = applyReadMeasurement(smu, U_read, read_aper, I_read);
    prev_R_after_last = R_r0;
    prev_G_after_last = 1 ./ R_r0;

    %% --- Main experiment loop ---
    for i = 1:N_total
        modeNow = ExpTable.Mode(i);
        U1      = ExpTable.Voltage(i);
        aper    = ExpTable.Duration(i) * 1e-3; % ms → s

        if modeNow == "set"
            % access device to set
            setDeviceN(arduinoObj,row_num,col_num)
            Icompl = I_set;
        else
            % access device to reset
            resetDeviceN(arduinoObj,row_num,col_num)
            Icompl = I_reset;
        end
        % Safety re-check: ensure compliance currents are within bounds
        if Icompl > 2e-4
            error("Compliance current too high (%.2e A) — aborting to protect device.", Icompl);
        end

        fprintf("Device %d/%d | Experiment %d/%d | Mode: %s | V=%.3f V | t=%.2f ms\n", ...
            d, numel(devices), i, N_total, modeNow, U1, ExpTable.Duration(i));

        % --- Store pre-experiment state ---
        TBL.State(i) = modeNow;
        TBL.V_applied(i) = U1;
        TBL.PulseDur_s(i) = aper;
        TBL.R_before(i) = prev_R_after_last;
        TBL.G_before(i) = prev_G_after_last;
        TBL.time_stamp(i) = string(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));

        % --- Apply main pulse ---
        [U_out, I1, R1, T1] = applyVoltage_single_voltages(smu, U1, read_aper, Icompl);
        TBL.U_out_immediate(i) = U_out;
        TBL.I_immediate(i)     = I1;
        TBL.R_immediate(i)     = R1;
        TBL.T_immediate(i)     = T1;

        % --- Read measurements at 1,2,...,10 s after pulse ---
        setDeviceN(arduinoObj,row_num,col_num)
        for t = read_times_s
            pause(1); % cumulative 1 s steps -> t seconds after the pulse
            [U_rt, I_rt, R_rt, T_rt] = applyReadMeasurement(smu, U_read, read_aper, I_read);
            TBL.(sprintf('R_after_%ds', t))(i) = R_rt;
            TBL.(sprintf('G_after_%ds', t))(i) = 1 ./ R_rt;
            TBL.(sprintf('I_after_%ds', t))(i) = I_rt;
            TBL.(sprintf('U_after_%ds', t))(i) = U_rt;
            TBL.(sprintf('T_after_%ds', t))(i) = T_rt;

            if t == read_times_s(end)
                % --- Carry forward last conductance ---
                prev_R_after_last = R_rt;
                prev_G_after_last = 1 ./ R_rt;
            end
        end

        % --- Save after each experiment ---
        if saveFile
            writetable(TBL, savepath);
        end

        pause(0.1);
    end

    %% --- Visualization ---
    if plotIV
        figure;
        scatter(TBL.V_applied, TBL.I_immediate, 30, (TBL.State=="reset"), 'filled');
        xlabel('Applied Voltage [V]');
        ylabel('Immediate Current [A]');
        title(sprintf('Set/Reset Experiments (row=%d, col=%d)', row_num, col_num));
        colormap([0.2 0.8 0.2; 0.9 0.2 0.2]);
        legend({'Set','Reset'});
    end

    fprintf('=== Finished device %d/%d: row=%d, col=%d ===\n', d, numel(devices), row_num, col_num);
end

turnOffSMUChannels(smu);

%% Abbreviations and column explanations:
% State: 'set' or 'reset' (experiment type, taken from the shared CSV design)
% ListIndex: index into the corresponding CSV (which row)
% V_applied_V: applied experiment pulse voltage (V)
% PulseDur_s: pulse duration in seconds
% U_out_immediate_V: measured voltage returned by SMU immediately after pulse
% I_immediate_A: measured current immediately after pulse
% R_immediate_Ohm: measured resistance immediately after pulse (SMU's R = V/I)
% R_before_Ohm, G_before_S: resistance and conductance measured before this experiment
% R_after_Ns_Ohm, G_after_Ns_S: resistance / conductance measured N seconds after pulse, for N = 1..10
% I_after_Ns_A: current measured at the N s read
% TimeStamp: timestamp for the start of the experiment row