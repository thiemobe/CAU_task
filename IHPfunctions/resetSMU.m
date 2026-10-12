function resetSMU(smu)
if strcmp('closed',get(smu,'status')), fopen(smu), end, %get(smu,'status'),
fwrite(smu,'*RST');  %fwrite(smu,':SYST:PRESet') %query(smu,'*cal?')
end