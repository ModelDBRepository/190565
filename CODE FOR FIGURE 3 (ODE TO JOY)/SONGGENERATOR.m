

%% Script converts network output into .wave file.  
clear ZS note;
QD = current';  %Need to use the current file.  
T1 = T/1000;
n = length(QD);                  % carrier frequency (Hz)
sf = 8192;
nt = T1*8192; 
Time = (1:1:nt)/8192; 

for i = 1:1:5 
QD(i,:) = (QD(i,:)-i)*2; 
ZS(i,:) = interp1((1:1:n)*T1/n,QD(i,:),Time); 
end

d = round(n/sf);                    % duration (s)             % number of samples
s = Time;          % sound data preparation


f(1) = 261; 
f(2) = 293.6;
f(3) = 329.6;
f(4) =  349.23;
f(5) = 392; 
j = 0; 
for ns = 1:1:5 
 note(ns,:) = sin(2*pi*f(ns)*Time); 
end

%%
songF = sum(ZS.*note);
q = length(songF);
%%
filename = 'SONG.wav'; 
audiowrite(filename,songF(0.9*q:q),sf);
