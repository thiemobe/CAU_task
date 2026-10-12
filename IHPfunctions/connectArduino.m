function arduinoObj = connectArduino
%CONNECTARDUINO connect arduino if it not done before
%   it return the arduinoObj (if necessary change COM number)
    arduinoObj = serialport('COM4', 57600);
end

