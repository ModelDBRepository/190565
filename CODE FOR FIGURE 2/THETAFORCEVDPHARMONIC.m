clear all
close all
clc

%% Fixed parameters across all simulations
dt = 0.00001; %time step 
N = 2000; %Network Size
td = 0.02; %decay time 
tr = 0.002; %Rise time 
%% 
T = 15; 
nt = round(T/dt);
tx = (1:1:nt)*dt; 

%% Van der Pol 
T = 15; nt = round(T/dt); 
mu = 0.5; %Sets the behavior  
MD = 10; %Scale system in space 
TC = 20; %Scale system in time 
[t,y]= ode45(@(t,y)vanderpol(mu,MD,TC,t,y),0:0.001:T,[0.1;0.1]); %run once to get to steady state oscillation
[t,y]= ode45(@(t,y)vanderpol(mu,MD,TC,t,y),0:0.001:T,y(end,:));
tx = (1:1:nt)*dt; 
xz(:,1) = interp1(t,y(:,1),tx); 
xz(:,2) = interp1(t,y(:,2),tx);
G = 10; Q = 10^4; T = 15; tmin = 5; tcrit = 10; 


%% 
m = min(size(xz)); %dimensionality of teacher 
E = Q*(2*rand(N,m)-1); %encoders
BPhi = zeros(N,m);  %Decoders
%% Compute Neuronal Intercepts and Tuning Curves 
initial = 0;
p = 0.1; %Sparse Coupling 
OMEGA = G*randn(N,N).*(rand(N,N)<p)/(sqrt(N)*p); %Random initial weight matrix 

%Set the sample row mean of the weight matrix to be 0 to strictly enforce
%balance.  
for i = 1:1:N 
    QS = find(abs(OMEGA(i,:))>0);
    OMEGA(i,QS) = OMEGA(i,QS) - sum(OMEGA(i,QS))/length(QS);
end

%% Storage Matrices and Initialization 
store = 10; %don't store every time step, saves time.  
current = zeros(nt,m);  %storage variable for output current 
IPSC = zeros(N,1); %post synaptic current 
h = zeros(N,1); r = zeros(N,1); hr = zeros(N,1); 
JD = 0*IPSC;

vpeak = pi; %peak and reset
vreset = -pi; 
v = vreset + (vpeak-vreset)*rand(N,1); %initialze voltage 
v_ = v; %temporary storage variable for integration 

j = 1;
time = zeros(round(nt/store),1);
RECB = zeros(5,round(2*round(nt/store)));
REC = zeros(10,round(nt/store));
tspike = zeros(8*nt,2);
ns = 0;
tic
SD = 0; 
BPhi = zeros(N,m);
z = zeros(m,1);
step = 50; %Sets the frequency of RLS  
imin = round(tmin/dt); %Start RLS
icrit = round((tcrit/dt)); %Stop RLS 

 Pinv = eye(N)*dt; 
 i = 1;
 %% 
 ilast = i; 
 %icrit = ilast;
for i = ilast :1:nt 
JX = IPSC + E*z; %current 
dv = 1-cos(v) + (1+cos(v)).*JX*(pi)^2;  %dv 
v = v_ + dt*(dv); %Euler integration plus refractory period.  
index = find(v>=vpeak);     
if length(index)>0
JD = sum(OMEGA(:,index),2); %compute the increase in current due to spiking  
tspike(ns+1:ns+length(index),:) = [index,0*index+dt*i];
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


v = v + (vreset - v).*(v>=vpeak); %reset with spike time interpolant implemented.  
v_ = v; 
 %only store stuff every index variable.  

 
 
 
z = BPhi'*r;
err = z - xz(i,:)';


if mod(i,step)==1
if i > imin 
 if i < icrit 
   cd = Pinv*r;    
   BPhi = BPhi - (cd*err');
   Pinv = Pinv - ((cd)*(cd'))/( 1 + (r')*(cd));
 end 
end 
end
  
 if mod(i,store) == 1;
        j = j + 1; 
time(j,:) = dt*i;        
current(j,:) = z; 
REC(:,j) = v(1:10); 
RECB(:,j) = BPhi(1:5,1);
    end


if mod(i,round(0.1/dt))==1

figure(1)
drawnow 
plot(tx(1:1:i),xz(1:1:i,:),'k','LineWidth',2), hold on
plot(time(1:1:j),current(1:1:j,:),'b--','LineWidth',2), hold off
xlim([dt*i-1,dt*i])
xlabel('Time')
ylabel('x(t)') 

figure(2)
plot(time(1:1:j),RECB(1:5,1:1:j),'.') 
xlabel('Time')
ylabel('\phi_j') 

figure(3)
plot(tspike(1:1:ns,2), tspike(1:1:ns,1),'k.')
ylim([0,100])
xlabel('Time')
ylabel('Neuron Index')
end

end
%% 
%ns
current = current(1:1:j,:);
REC = REC(:,1:1:j);
%REC2 = REC2(:,1:1:j);
nt = length(current);
time = (1:1:nt)*T/nt; 
xhat = current; 
tspike = tspike(1:1:ns,:); 
M = tspike(tspike(:,2)>dt*icrit,:);
AverageFiringRate = length(M)/(N*(T-dt*icrit))
%% 
Z = eig(OMEGA); %eigenvalues pre-learning
Z2 = eig(OMEGA+E*BPhi');   %eigenvalues post-learning
figure(42)
plot(Z,'k.'), hold on 
plot(Z2,'r.')
xlabel('Re\lambda')
ylabel('Im\lambda') 
legend('Pre-Learning','Post-Learning')

%% Plot neurons pre- and post- learning
figure(43)
for z = 1:1:10 
plot((1:1:j)*T/j,(REC(z,1:1:j))/(2*pi)+z), hold on    
end
xlim([9,10])
xlabel('Time (s)')
ylabel('Neuron Index')
title('Post Learning')
figure(66)
for z = 1:1:10 
plot((1:1:j)*T/j,(REC(z,1:1:j))/(2*pi)+z), hold on    
end
xlim([0,1])
title('Pre-Learning')
xlabel('Time (s)')
ylabel('Neuron Index')