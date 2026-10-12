function [] = turnOnSMUChannels(smu)
%UNTITLED turn On SMU Channels to start measurements
%   sour2 - set 2.5 Volt output (voltage supply)
fwrite(smu,':sour2:func:mode volt');                      % query(smu,':sour2:func:mode?'),
fwrite(smu,':sour2:volt 2.5');                            % query(smu,':sour2:volt?'),

fwrite(smu,':OUTP1 ON; OUTP2 ON');

end

