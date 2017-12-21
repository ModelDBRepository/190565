clear all
clc 

%%
N = 2000;  %Number of neurons 
dt = 0.00005;
tref = 0.002; %Refractory time constant in seconds 
tm = 0.01; %Membrane time constant 
vreset = -65; %Voltage reset 
vpeak = -40; %Voltage peak. 
td = 0.02; tr = 0.002;
%% 
 
 alpha = dt*0.05 ; %Sets the rate of weight change, too fast is unstable, too slow is bad as well.  
 Pinv = eye(N)*alpha; %initialize the correlation weight matrix for RLS
 p = 0.1; %Set the network sparsity 

%% Target Dynamics for Product of Sine Waves
T = 15; imin = round(5/dt); icrit = round(10/dt); step = 50; nt = round(T/dt); Q = 30; G = 0.1;
zx = (sin(2*pi*(1:1:nt)*dt*5));

 

%%
k = min(size(zx));
IPSC = zeros(N,1); %post synaptic current storage variable 
h = zeros(N,1); %Storage variable for filtered firing rates
r = zeros(N,1); %second storage variable for filtered rates 
hr = zeros(N,1); %Third variable for filtered rates 
JD = 0*IPSC; %storage variable required for each spike time 
tspike = zeros(4*nt,2); %Storage variable for spike times 
ns = 0; %Number of spikes, counts during simulation  
z = zeros(k,1);  %Initialize the approximant 
 
v = vreset + rand(N,1)*(30-vreset); %Initialize neuronal voltage with random distribtuions
v_ = v;  %v_ is the voltage at previous time steps  
RECB = zeros(nt,10);  %Storage matrix for the synaptic weights (a subset of them) 
OMEGA =  G*(randn(N,N)).*(rand(N,N)<p)/(sqrt(N)*p); %The initial weight matrix with fixed random weights  
BPhi = zeros(N,k); %The initial matrix that will be learned by FORCE method

%set the row average weight to be zero, explicitly.
for i = 1:1:N 
    QS = find(abs(OMEGA(i,:))>0);
    OMEGA(i,QS) = OMEGA(i,QS) - sum(OMEGA(i,QS))/length(QS);
end


 E = (2*rand(N,k)-1)*Q;
REC2 = zeros(nt,20);
REC = zeros(nt,10);
current = zeros(nt,k);  %storage variable for output current/approximant 
i = 1; 















tlast = zeros(N,1); %This vector is used to set  the refractory times 
BIAS = vpeak; %Set the BIAS current, can help decrease/increase firing rates.  
%%
ilast = i; 
%icrit = ilast;
for i = ilast:1:nt 

     
I = IPSC + E*z + BIAS; %Neuronal Current 
 
dv = (dt*i>tlast + tref).*(-v+I)/tm; %Voltage equation with refractory period 
v = v + dt*(dv);

index = find(v>=vpeak);  %Find the neurons that have spiked 


%Store spike times, and get the weight matrix column sum of spikers 
if length(index)>0
JD = sum(OMEGA(:,index),2); %compute the increase in current due to spiking  
tspike(ns+1:ns+length(index),:) = [index,0*index+dt*i];
ns = ns + length(index);  % total number of psikes so far
end

tlast = tlast + (dt*i -tlast).*(v>=vpeak);  %Used to set the refractory period of LIF neurons 

% Code if the rise time is 0, and if the rise time is positive 
if tr == 0  
    IPSC = IPSC*exp(-dt/td)+   JD*(length(index)>0)/(td);
    r = r *exp(-dt/td) + (v>=vpeak)/td;
else
    IPSC = IPSC*exp(-dt/tr) + h*dt;
h = h*exp(-dt/td) + JD*(length(index)>0)/(tr*td);  %Integrate the current

r = r*exp(-dt/tr) + hr*dt; 
hr = hr*exp(-dt/td) + (v>=vpeak)/(tr*td);
end



%Implement RLS with the FORCE method 
 z = BPhi'*r; %approximant 
 err = z - zx(:,i); %error 
 %% RLS 
 if mod(i,step)==1 
if i > imin 
 if i < icrit 
   cd = Pinv*r;
   BPhi = BPhi - (cd*err');
   Pinv = Pinv -((cd)*(cd'))/( 1 + (r')*(cd));
 end 
end 
 end

v = v + (30 - v).*(v>=vpeak);

REC(i,:) = v(1:10); %Record a random voltage 

v = v + (vreset - v).*(v>=vpeak); %reset with spike time interpolant implemented.  
current(i,:) = z;
RECB(i,:) = BPhi(1:10);  
REC2(i,:) = r(1:20); 



    if mod(i,round(0.5/dt))==1
    dt*i
  drawnow
figure(1)
plot(tspike(1:1:ns,2),tspike(1:1:ns,1),'k.')
xlim([dt*i-5,dt*i])
ylim([0,200])
figure(2)
plot(dt*(1:1:i),zx(:,1:1:i),'k--','LineWidth',2), hold on
plot(dt*(1:1:i),current(1:1:i,:),'LineWidth',2), hold off

xlim([dt*i-5,dt*i])
%ylim([-0.5,0.5])

%xlim([dt*i,dt*i])
figure(5)
plot(dt*(1:1:i),RECB(1:1:i,1:10),'.')
    end
end
time = 1:1:nt; 
%% 
TotNumSpikes = ns 
%tspike = tspike(1:1:ns,:); 
M = tspike(tspike(:,2)>dt*icrit,:); 
AverageRate = length(M)/(N*(T-dt*icrit)) 

%% Plotting 
figure(30)
for j = 1:1:5
plot((1:1:i)*dt,REC(1:1:i,j)/(30-vreset)+j), hold on 
end
xlim([T-2,T])
xlabel('Time (s)')
ylabel('Neuron Index') 
title('Post Learning')
figure(31)
for j = 1:1:5
plot((1:1:i)*dt,REC(1:1:i,j)/(30-vreset)+j), hold on 
end
xlim([0,2])
xlabel('Time (s)')
ylabel('Neuron Index') 
title('Pre-Learning')
%% 
Z = eig(OMEGA+E*BPhi'); %eigenvalues after learning 
Z2 = eig(OMEGA);  %eigenvalues before learning 
%% Plot eigenvalues before and after learning
figure(40)
plot(Z2,'r.'), hold on 
plot(Z,'k.') 
legend('Pre-Learning','Post-Learning')
xlabel('Re \lambda')
ylabel('Im \lambda')
