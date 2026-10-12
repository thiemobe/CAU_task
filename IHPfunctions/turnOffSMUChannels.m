function [] = turnOffSMUChannels(smu)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

fwrite(smu,':OUTP1 OFF; OUTP2 OFF');

end

