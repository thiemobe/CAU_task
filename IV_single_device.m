
% fopen(smu)%Choose memristor's row and colomn
%%
row_num = 5;
col_num = 5;

%Plot IV?
plotIV = true;
% plotIV = false;

%Create a file? (filename sould be new)
% saveFile = true;
saveFile = false;

addpath("IHPfunctions\")

%Connect arduino (if necessary change COM number)
if ~exist('arduinoObj')
    arduinoObj = connectArduino;
end

%Check visa address SMU 
if ~exist('smu')
    smu = connectSMU;
end

% resetSMU(smu);

if saveFile
file=['IV_' row_str col_str]; 
current_path= pwd;
    if isfile(fullfile(current_path, append(file, '.txt')))==1;
        'File already exists'
        return;
    end
end


nc=2;               % number of sweep_cycles
U1peak1=1;         % sweep voltage set [+V]
U1peak2=-1.5;        % sweep voltage reset [-V]
Ustep=0.1;          % sweep voltage step [V]
I1compl=1e-4;       % I_compliance set [A]
I1comp2=2*I1compl;  % I_compliance reset [A]
aper=100e-3;          % normal time [s]


turnOnSMUChannels(smu)

IV_main

turnOffSMUChannels(smu)


%save file
if saveFile
    writematrix(IV,[file,'.txt']);
end
%%

%plot data
if plotIV
    figure
    plot(IV(:,2),IV(:,3))
    xlabel('Voltage [V]')
    ylabel('Current [A]')
end
%%
% histogram(IV(2:80:end, 4))
% histogram(IV(40:80:end, 4))
% std(IV(2:80:end, 4))/mean(IV(2:80:end, 4))
% std(IV(40:80:end, 4))/mean(IV(40:80:end, 4))
