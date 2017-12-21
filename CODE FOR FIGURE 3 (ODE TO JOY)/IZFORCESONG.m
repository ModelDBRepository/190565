% Network of Izhikevich Neurons learns the first bar of Ode to Joy 
% song note data is located in the file ode2joy.mat. 

clear all
clc 

T = 1000000;  %Network parameters and total simulation time 
dt = 0.04; %Time step 
nt = round(T/dt);  %number of time steps 
N =  5000; %number of neurons 
%% Izhikevich Parameters
C = 250;
vr = -60; 
b = 0; 
ff = 2.5; 
vpeak = 30; 
vreset = -65;
vt = vr+40-(b/ff); 
Er = 0;
u = zeros(N,1); 
a = 0.01;
d = 200; 
tr = 2; 
td = 20; 
p = 0.1; 
G =1*10^4; %Controls the magnitude of chaos
Q = 4*10^3; %Controls the magnitude of the perturbation.  
%%  Initialize post synaptic currents, and voltages 
IPSC = zeros(N,1); %post synaptic current 
h = zeros(N,1);
r = zeros(N,1);
hr = zeros(N,1);
JD = zeros(N,1);

v = vr+(vpeak-vr)*rand(N,1); %initial distribution 
v_ = v; %These are just used for Euler integration, previous time step storage


%% Convert the sequence of notes and half notes in the ode2joyshort.mat into a teaching signal. 
% the file ode2joy short.mat contains 2 matrices, J and HN.  The matrix
% length corresponds to the number of notes while the matrix width
% corresponds to the note type and is the dimension of the teaching signal.
% J indicates the presence of a note while HN indicates the presence of a
% half note.  
freq = 4; 
load ode2joyshort.mat;
nnotes = length(J);
nchord = min(size(J));
ds = (1000/freq)*nnotes; n1 = round(ds/dt);
ZS = abs(sin(pi*(1:1:n1)*dt*nnotes/(ds)));
ZS = repmat(ZS,nchord,1);
song = J'; 
nn = size(song);  

j = 1 ;
for i = 1:1:n1 
    if mod(i,round(1000/(freq*dt)))==0;
    j = j + 1;
    if j > nn(2); break; end 
    end
    ZS(:,i) = ZS(:,i).*song(:,j);
end 

for i = 1:1:15;
    if HN(i,1) > 0 
        q = length((i-1)*(1000/(freq*dt)):(i+1)*(1000/(freq*dt)));
        w = find(J(i,:)>0);
       ZS(w,(i-1)*(1000/(freq*dt)):(i+1)*(1000/(freq*dt)))= sin(pi*(1:1:q)/q);
    end 
end
zx = repmat(ZS,1,ceil(T/ds));





%%
E = (2*rand(N,nchord)-1)*Q;  %Rank-nchord perturbation 
OMEGA =  G*(randn(N,N)).*(rand(N,N)<p)/(p*sqrt(N)); %Random weight matrix 
z = zeros(nchord,1);  %initialize approximant  
BPhi = zeros(N,nchord); %initialize Decoder
tspike = zeros(nt,2); %spike times 
ns = 0;
BIAS = 1000; %Bias, at the rheobase current.  

%%
 Pinv = eye(N)*2;  %The initial correlation weight matirx  
 step = 100; %Total number of steps to use 
 imin = round(1000/dt); %First step to start RLS/FORCE method 
 icrit = round(0.9*T/dt); %Last step to start RLS/FORCE method 
 current = zeros(nt,nchord); %Store the approxoimant 
 RECB = zeros(nt,5); %Store some decoders 
 REC = zeros(nt,10); %Store some voltage traces/adaptation parameters
 i=1; ss = 0;
%% SIMULATION
tic
ilast = i ;
%icrit = ilast;
 
for i = ilast:1:nt; 
%% EULER INTEGRATE
I = IPSC + E*z  + BIAS; 
v = v + dt*(( ff.*(v-vr).*(v-vt) - u + I))/C ; % v(t) = v(t-1)+dt*v'(t-1)
u = u + dt*(a*(b*(v_-vr)-u)); %same with u, the v_ term makes it so that the integration of u uses v(t-1), instead of the updated v(t)
%% 
index = find(v>=vpeak);
if length(index)>0
JD = sum(OMEGA(:,index),2); %compute the increase in current due to spiking  
%tspike(ns+1:ns+length(index),:) = [index,0*index+dt*i]; %Store spike
%times, but takes longer to simulate.  
ns = ns + length(index);  %total number of spikes 
end

% implement the synapse, either single or double exponential
if tr == 0 
    IPSC = IPSC*exp(-dt/td)+   JD*(length(index)>0)/(td);
    r = r *exp(-dt/td) + (v>=vpeak)/td;
else
IPSC = IPSC*exp(-dt/tr) + h*dt;
h = h*exp(-dt/td) + JD*(length(index)>0)/(tr*td);  %Integrate the current

r = r*exp(-dt/tr) + hr*dt; 
hr = hr*exp(-dt/td) + (v>=vpeak)/(tr*td);
end

%Compute the approximant and error 
 z = BPhi'*r;
 err = z - zx(:,i);
 zz(:,i) = zx(:,i) + cumsum(ones(nchord,1));
 
 %% Implement RLS.  
 if mod(i,step)==1 
if i > imin 
 if i < icrit 
   cd = Pinv*r;
   BPhi = BPhi - (cd*err');
   Pinv = Pinv -((cd)*(cd'))/( 1 + (r')*(cd));
 end 
end 
 end

 %Record the decoders periodically.  
if mod(i,1/dt)==1
    ss = ss + 1; 
    RECB(ss,:)=BPhi(1:5);
end


% apply the resets and store stuff 
u = u + d*(v>=vpeak);  %implements set u to u+d if v>vpeak, component by component. 
v = v+(vreset-v).*(v>=vpeak); %implements v = c if v>vpeak add 0 if false, add c-v if true, v+c-v = c
v_ = v;  % sets v(t-1) = v for the next itteration of loop
REC(i,:) = [v(1:5)',u(1:5)']; 
current(i,:) = z'+ cumsum(ones(1,nchord)); 

%Plot progress 
if mod(i,round(100/dt))==1 
drawnow
%figure(1)
%plot(tspike(1:1:ns,2),tspike(1:1:ns,1),'k.')
%ylim([0,100])
gg = max(1,i-round(3000/dt));

figure(2)
plot(dt*(gg:1:i)/1000,current(gg:1:i,:)), hold on 
plot(dt*(gg:1:i)/1000,zz(:,gg:1:i),'k--'), hold off
xlim([dt*i/1000-3,dt*i/1000])
xlabel('Time (s)')
ylabel('Note') 
figure(5)
plot(0.001*dt*i*(1:1:ss)/ss,RECB(1:1:ss,:),'.'), hold off 
xlabel('Time (s)')
ylabel('Decoder')

end   

end
