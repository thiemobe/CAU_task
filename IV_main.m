U1=[]; I1=[]; R1=[]; To=[]; IV = [];

for i=1:nc
    i
    %set part of IV
    if abs(U1peak1)>= 1e-3                % U1peak1 ~= 0
        % access device to set
        setDeviceN(arduinoObj,row_num,col_num)

        U1=[0:Ustep:U1peak1 U1peak1-Ustep:-Ustep:Ustep];
        [U_out,I1,R1,T] = applyVoltage(smu, U1, aper, I1compl);

        To = [U1', U_out', I1', R1', T'];
        IV = [IV; To];
    end
    
    %reset part of IV
    if abs(U1peak2) >= 1e-3
        % access device to reset
        resetDeviceN(arduinoObj,row_num,col_num)

        U1=[0:Ustep:-U1peak2 -U1peak2-Ustep:-Ustep:Ustep];
        [U_out,I1,R1,T] = applyVoltage(smu, U1, aper, I1comp2);
        IV = [IV; To];

    end
end