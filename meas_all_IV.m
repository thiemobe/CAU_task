%Connect arduino (if necessary change COM number)
if ~exist('arduinoObj')
    arduinoObj = serialport('COM13', 57600);
end

%Folder name (change the name if you have change IC)
dirName = 'IV_curves_all';
mkdir IV_curves_all

%Data storage name (change the name if you have change IC)
load('crossbarMap.mat')

if ~exist('HRS2LRS')
	HRS2LRS(1:64,1:64) = 0;
end
if ~exist('Ron')
	Ron(1:64,1:64) = 0;
end
if ~exist('Roff')
	Roff(1:64,1:64) = 0;
end

%IV parameters
nc=1;               % number of sweep_cycles
U1peak1=2.;        % sweep voltage set [+V]
U1peak2=-2.;       % sweep voltage reset [-V]
Ustep=0.1;          % sweep voltage step [V]
I1compl=1e-4;       % I_compliance set [A]
I1comp2=2*I1compl;    % I_compliance reset [A]
aper=100e-3;          % normal time [ms]
Uread=0.1;          % read voltage [V]

round((2*U1peak1+Uread)/Ustep + 1)


if ~exist('smu'),
 global smu
 %smu = visa('AGILENT', 'USB0::0x0957::0x8C18::MY51143492::0::INSTR');
 smu = visa('AGILENT', 'USB0::0x0957::0x8C18::MY51143492::0::INSTR');
%  USB0::0x0957::0x8C18::MY51143492::0::INSTR
 smu.InputBufferSize = 100000;   %  8388608
 smu.OutputBufferSize= 200000;
 smu.Timeout = 1*60; % in sec
end

if strcmp('closed',get(smu,'status')), fopen(smu), end, %get(smu,'status'),
fwrite(smu,'*RST');  %fwrite(smu,':SYST:PRESet') %query(smu,'*cal?')

% sour2 - set 2.5 Volt output (voltage supply)
fwrite(smu,':sour2:func:mode volt');                      % query(smu,':sour2:func:mode?'),
fwrite(smu,':sour2:volt 2.5');                            % query(smu,':sour2:volt?'),

fwrite(smu,':OUTP1 ON; OUTP2 ON');

tic
for row_num=1:64
    row_num
    for col_num = 1:64
%         % condition to check IV
%         if HRS2LRS(row_num,col_num)>=0 
            IV_map;
            dlmwrite(string(pwd)+'\'+dirName+'\IV_'+sprintf('%02i',row_num)+sprintf('%02i',col_num)+'.txt',IV,',');
            Ron(row_num,col_num) = IV(round((2*U1peak1+Uread)/Ustep + 1),4);
            Roff(row_num,col_num) = IV(end+1-round(Uread/Ustep),4);
            HRS2LRS(row_num,col_num) = Roff(row_num,col_num)/Ron(row_num,col_num);
%             % condition to check if the value is real         
            if Ron(row_num,col_num)<0 || Ron(row_num,col_num)>1e8 || Roff(row_num,col_num)<0 || Roff(row_num,col_num)>1e8 
                Ron(row_num,col_num) = 0;
                Roff(row_num,col_num) = 0;
                HRS2LRS(row_num,col_num) = 0;
            end
%         end
    end
end
toc
 
save  crossbarMap HRS2LRS Ron Roff

fwrite(smu,':OUTP1 OFF; OUTP2 OFF');
%fclose(smu);
%%
% figure()
% imagesc(HRS2LRS)
% colorbar

% load('crossbarMap.mat')

figure()
HRS2LRS2fit = HRS2LRS(HRS2LRS>5);
h1 = histogram(HRS2LRS2fit)
xlabel('Resistance ratio')
ylabel('Device number')
 morebins(h1)
 
figure()
Gon2fit=1./Ron(HRS2LRS>5);
h2 = histogram(Gon2fit)
legend('LRS')
% xlabel('Resistance [Ohm]')
xlabel('Conductance [S]')
ylabel('Device number')

figure()
Goff2fit=1./Roff(HRS2LRS>5);
h3 = histogram(Goff2fit)
 morebins(h3)
legend('HRS')
% xlabel('Resistance [Ohm]')
xlabel('Conductance [S]')
ylabel('Device number')