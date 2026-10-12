 function [] = setDeviceN(arduinoObj, row_num, col_num)
%SETDEVICEN access device N row_num, col_num to set

%Row and colomn to string format XX
row_str =['0' int2str(row_num)];
row_str = row_str(end-1:end);

col_str =['0' int2str(col_num)];
col_str = col_str(end-1:end);

% write the command to arduino
cmd = [row_str col_str 's'];
writeline(arduinoObj,cmd);

end

