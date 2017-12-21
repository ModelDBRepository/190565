% Network of Izhikevich Neurons learns a song bird signal with a clock
% input.  Note that you have to supply your own supervisor here due to file
% size limitations.  The supervisor, zx should be a matrix of  m x nt dimensional, where
% m is the dimension of the supervisor and nt is the number of time steps.
% RLS is applied until time T/2.  The HDTS is stored as variable z2.  Note
% that the code is written for an 8 second supervisor, nt should equal the
% length of z2.

clear all
clc 

T = 100000;  %Network parameters and total simulation time 
dt = 0.04; %Time step 
nt = round(T/dt);  %number of time steps 
N =  1000; %number of neurons 
%% Izhikevich Parameters
C = 250;
vr = -60; 
b = 0; 
ff = 2.5; 
vpeak = 30; 
vreset = -65;
vt = -40;
Er = 0;
u = zeros(N,1); 
a = 0.002;
d = 100; 
tr = 2; 
td = 20; 
p = 0.1; 
G = 5*10^3;
Q = 4*10^2; 
WE2 = 4*10^3;

%%  Initialize post synaptic currents, and voltages 
IPSC = zeros(N,1); %post synaptic current 
h = zeros(N,1);
r = zeros(N,1);
hr = zeros(N,1);
JD = zeros(N,1);

v = vr+(vpeak-vr)*rand(N,1); %initial distribution 
v_ = v; %These are just used for Euler integration, previous time step storage

%% User has to supply supervisor signal, zx
zx = vz;
dd = size(zx); 
m1 = dd(1);
m2 = 32; %number of upstates in the supervisor signal duration of 5 seconds.  100 per second.  
zx = zx/(max(max(zx)));
zx(isnan(zx)==1)=0; 
%% generate HDTS signal  
temp1 = abs(sin(m2*pi*((1:1:8000/dt)*dt)/8000));
for qw = 1:1:m2
z2(qw,:) = temp1.*((1:1:8000/dt)*dt<qw*8000/m2).*((1:1:8000/dt)*dt>(qw-1)*8000/m2);
end
%%
dd(2) = max(size(zx));
OMEGA =  G*(randn(N,N)).*(rand(N,N)<p)/(p*sqrt(N)); %Random weight matrix 
z1 = zeros(m1,1); 
BPhi1 = zeros(N,m1); %initialize Decoder
E1 = (2*rand(N,m1)-1)*Q;  %Rank-nchord perturbation 
E2 = (2*rand(N,m2)-1)*WE2; 

tspike = zeros(8*nt,2); %spike times 
ns = 0;
BIAS = 1000; %Bias, at the rheobase current.  

%%
 Pinv1 = eye(N)*2;  %The initial correlation weight matirx  
 step = 100; %Total number of steps to use 
 imin = round(1000/dt); %First step to start RLS/FORCE method 
 icrit = round(0.5*T/dt); %Last step to start RLS/FORCE method 
 current = zeros(nt/100,m1); %Store the approxoimant 
 RECB = zeros(nt,10); %Store some decoders %
 REC = zeros(nt,10); %Store some voltage traces 
 i=1; ss = 0;
 qq = 1; 
 k2 = 0; 
 ns1 = 0; ns2 = 0; 
%% SIMULATION
tic
ilast = i ;
%icrit = ilast;

for i = ilast:1:nt; 
%% EULER INTEGRATE
I = IPSC + E1*z1 + E2*z2(:,qq) +  BIAS; 
v = v + dt*(( ff.*(v-vr).*(v-vt) - u + I))/C ; % v(t) = v(t-1)+dt*v'(t-1)
u = u + dt*(a*(b*(v_-vr)-u)); %same with u, the v_ term makes it so that the integration of u uses v(t-1), instead of the updated v(t)
%% 
index = find(v>=vpeak);
if length(index)>0
JD = sum(OMEGA(:,index),2); %compute the increase in current due to spiking  
tspike(ns+1:ns+length(index),:) = [index,0*index+dt*i]; %Store spike
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
 z1 = BPhi1'*r;
 if qq>=dd(2)
     qq = 1;
 end
 err = z1 - zx(:,qq);
qq = qq + 1; 
 
 %% Implement RLS.  
 if mod(i,step)==1 
if i > imin 
 if i < icrit 

   cd1 = Pinv1*r;
   BPhi1 = BPhi1 - (cd1*err');
   Pinv1 = Pinv1 -((cd1)*(cd1'))/( 1 + (r')*(cd1));
 end 
end 
 end

 %Record the decoders periodically.  
if mod(i,1/dt)==1
    ss = ss + 1; 
    RECB(ss,1:10)=BPhi1(1:10);
end


% apply the resets and store stuff 
u = u + d*(v>=vpeak);  %implements set u to u+d if v>vpeak, component by component. 
v = v+(vreset-v).*(v>=vpeak); %implements v = c if v>vpeak add 0 if false, add c-v if true, v+c-v = c
v_ = v;  % sets v(t-1) = v for the next itteration of loop
REC(i,:) = [v(1:5)',u(1:5)']; 
 if mod(i,100) ==1 
     k2 = k2 + 1;
 current(k2,:) = z1'; 
 end
%Plot progress 
if mod(i,round(100/dt))==1 
dt*i
drawnow
%figure(1)
%plot(tspike(1:1:ns,2),tspike(1:1:ns,1),'k.')
%ylim([0,100])
gg = max(1,i-round(3000/dt));

figure(32)
plot(0.001*(1:1:k2)*dt*i/k2,current(1:1:k2,1:20:m1)) 
xlabel('Time')
ylabel('Network Output')
xlim([dt*i/1000-5,dt*i/1000])
ylim([0,0.4])

figure(5)
plot(0.001*dt*i*(1:1:ss)/ss,RECB(1:1:ss,1:10),'r.')
xlabel('Time (s)')
ylabel('Decoder')

end   

end
