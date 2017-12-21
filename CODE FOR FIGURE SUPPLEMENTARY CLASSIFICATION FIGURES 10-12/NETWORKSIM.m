%% IZFORCE WITH ADAPTATION, Load a Pre-trained weight matrix and test data.   
clear all
clc 
%%
T = 10000;
dt = 0.04;
nt = round(T/dt);
N =  2000;   %number of neurons 
load WEIGHTCLASSLINEAR.mat   %Load trained weight data 
%% Izhikevich Parameters
C = 250;
vr = -60; 
b = 0; 
k = 2.5; 
vpeak = 30; 
vreset = -65;
vt = vr+40-(b/k); %threshold 
Er = 0; %Reversal Potential 
u = zeros(N,1); 
a = 0.01;
d = 200; 
tr = 2; 
td = 20; 
p = 0.1; 

%% Initalize currents all to be 0
IPSC = zeros(N,1); %post synaptic current 
h = zeros(N,1);
r = zeros(N,1);
hr = zeros(N,1);
JD = zeros(N,1);


%-----Initialization---------------------------------------------
v = vr+(vpeak-vr)*rand(N,1); %initial distribution 
v_ = v; %These are just used for Euler integration, previous time step storage


%% 
% Load a test data set for classification. 
load testsetlinear.mat;  
inputfreq = 4; %Present a data point at a specific frequency, 4 Hz.  
Xin = zeros(nt,2); 
zx = zeros(nt,1); 
nx = round(1000/(dt*inputfreq)); 
zx = 0.5*abs(sin(2*pi*(1:1:nt)*dt*inputfreq/2000));
 

%% Construct input and correct response.  
j = 1; 
k2 = 0;
for i =1:1:nt 
 zx(i) = zx(i)*(2*P(j)-1)*mod(k2,2);
 Xin(i,:) = [z(j,1),z(j,2)]*mod(k2,2);
if mod(i,nx)==1 
    k2 = k2 + 1; 
j = ceil(rand*2000);
end
end
clear z P






%% initialize network output and spike times.  
z = 0; 
tspike = zeros(nt,2);
ns = 0;


%% 
 current = zeros(nt,1); 
 REC = zeros(nt,10);
 i=1;
%% SIMULATION
tic
ilast = i ;
for i = ilast:1:nt; 
%% EULER INTEGRATE
I = IPSC + E*z + Ein*(Xin(i,:)') + BIAS; 
v = v + dt*(( k.*(v-vr).*(v-vt) - u + I))/C ; % v(t) = v(t-1)+dt*v'(t-1)
u = u + dt*(a*(b*(v_-vr)-u)); %same with u, the v_ term makes it so that the integration of u uses v(t-1), instead of the updated v(t)

%% 
index = find(v>=vpeak);
if length(index)>0
JD = sum(OMEGA(:,index),2); %compute the increase in current due to spiking  
%tspike(ns+1:ns+length(index),:) = [index,0*index+dt*i];
ns = ns + length(index); 
end

if tr == 0 
    IPSC = IPSC*exp(-dt/td)+   JD*(length(index)>0)/(td);
    r = r *exp(-dt/td) + (v>=vpeak)/td;
else
IPSC = IPSC*exp(-dt/tr) + h*dt;
h = h*exp(-dt/td) + JD*(length(index)>0)/(tr*td);  %Integrate the current

r = r*exp(-dt/tr) + hr*dt; 
hr = hr*exp(-dt/td) + (v>=vpeak)/(tr*td);
end


 z = BPhi'*r;


%% COMPUTE S, APPLY RESETS
u = u + d*(v>=vpeak);  %implements set u to u+d if v>vpeak, component by component. 
v = v+(vreset-v).*(v>=vpeak); %implements v = c if v>vpeak add 0 if false, add c-v if true, v+c-v = c
v_ = v;  % sets v(t-1) = v for the next itteration of loop
REC(i,:) = [v(1:5)',u(1:5)']; 
current(i,:) = z; 
if mod(i,round(100/dt))==1 
drawnow
figure(2)
plot(dt*(1:1:i),current(1:1:i,1)), hold on 
plot(dt*(1:1:i),zx(1:1:i),'k--'), hold off
ylim([-0.6,0.6])
xlim([dt*i-3000,dt*i])
xlabel('Time')
ylabel('Network Response')
legend('Network','Correct Response')

end   



end
AverageFiringRate = 1000*ns/(N*T)
