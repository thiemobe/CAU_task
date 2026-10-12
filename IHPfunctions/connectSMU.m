function smu = connectSMU
%CONNECTSMU connect SMU with parameters below
   %Check visa address SMU 
 smu = visa('AGILENT', 'USB0::0x2A8D::0x9201::MY63321087::0::INSTR');
 smu.InputBufferSize = 100000;   %  8388608
 smu.OutputBufferSize = 200000;
 smu.Timeout = 1*60; % in sec
end

