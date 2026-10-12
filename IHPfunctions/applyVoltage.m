 function [U_out,I1,R1,T] = applyVoltage(smu, U1, aper, I1compl)
%APPLYVOLTAGE apply voltage U1 with timestep aper and current compl I1compl

I1=[]; R1=[];




np = size (U1,2);
list=[num2str(U1(1:(np-1)),'%10.6e,'),num2str(U1(np),'%10.6e')];


fwrite(smu,':sour:func:mode volt');
fwrite(smu,[':sour:volt:mode list;:sour:list:volt ',list]); % list1=query(smu,':sour:list:volt?'),

% sense
fwrite(smu,':sens:func "curr","res"');              % query(smu,':sens:func?')
fwrite(smu,':sens:curr:rang:auto on');
%fwrite(smu,'sens:curr:dc:rang:upp 1e-6');          % query(smu,':sens:curr:dc:rang:upp?')
% fwrite(smu,[':sens:curr:dc:aper ',num2str(aper)]);  % aper=str2num(query(smu,':sens:curr:dc:aper?')),
fwrite(smu,[':sens:curr:prot:pos ',num2str(I1compl)]);  % i1compl=str2num(query(smu,':SENS1:CURR:PROT?'))
% fwrite(smu,[':sens:curr:prot:neg ',num2str(I1comp2)]);

% fwrite(smu,':trig:sour aint'); fwrite(smu,[':trig:coun ',num2str(nc*np)]); % triger
fwrite(smu,':trig:sour aint'); fwrite(smu,[':trig:coun ',num2str(np)]); % triger  
fprintf(smu, strjoin([":TRIG:COUN ", num2str(length(U1))]));
fprintf(smu, ":TRIG:SOUR TIM");
fprintf(smu, strjoin([":TRIG:TIM ", aper]));
fwrite(smu,':disp:enab on;:disp:view grap'), % grafic display
%fwrite(smu,':disp:enab on;:disp:view sing1'), % single1 display

% If=str2num(query(smu,'fetc:scal:curr? (@1)'));disp(If)
%start measurement
%tic
%  fwrite(smu,':outp1 on');
 fwrite(smu,':init (@1)');
 opc=query(smu, '*opc?');
%  fwrite(smu,':outp1 off');
 
%toc

%retrieve measurement result

I1=str2num(query(smu,'fetc:arr:curr? (@1)'));% I1=reshape(I1,np,nc); U1=U1(:);
R1=str2num(query(smu,'fetc:arr:res? (@1)')); %R1=reshape(R1,np,nc); %figure, plot(U1,R1), grid,
U_out=str2num(query(smu,'fetc:arr:volt? (@1)')); %U_out=reshape(U_out,np,nc);
T=str2num(query(smu,'fetc:arr:time? (@1)')); % T=reshape(T,np,nc);
 
% To=[];
% % for p=1:nc
% To=[To, Uread, U_out(:,p), I1(:,p), R1(:,p), T(:,p)];
% % end

end

