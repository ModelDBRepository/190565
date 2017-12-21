%% Force Method with Izhikevich Network 
clear all
close all
clc 

T = 50000; %Total time in ms
dt = 0.04; %Integration time step in ms 
nt = round(T/dt); %Time steps
N =  2000;  %Number of neurons 
%% Izhikevich Parameters
C = 250;  %capacitance
vr = -60;   %resting membrane 
b = -2;  %resonance parameter 
ff = 2.5;  %k parameter for Izhikevich, gain on v 
vpeak = 30;  % peak voltage
vreset = -65; % reset voltage 
vt = vr+40-(b/ff); %threshold  %threshold 
u = zeros(N,1);  %initialize adaptation 
a = 0.01; %adaptation reciprocal time constant 
d = 200; %adaptation jump current 
tr = 2;  %synaptic rise time 
td = 20; %decay time 
p = 0.1; %sparsity 
G =1*10^4; %Gain on the static matrix with 1/sqrt(N) scaling weights.  
Q =1*10^4; %Gain on the rank-k perturbation modified by RLS.  
Irh = 0.25*ff*(vt-vr)^2; 

%Storage variables for synapse integration  
IPSC = zeros(N,1); %post synaptic current 
h = zeros(N,1); 
r = zeros(N,1);
hr = zeros(N,1);
JD = zeros(N,1);

%-----Initialization---------------------------------------------
v = vr+(vpeak-vr)*rand(N,1); %initial distribution 
v_ = v; %These are just used for Euler integration, previous time step storage

%% Target signal  COMMENT OUT TEACHER YOU DONT WANT, COMMENT IN TEACHER YOU WANT. 
zx =  sin(2*pi*(1:1:nt)*dt*2/1000).*sin(2*pi*(1:1:nt)*dt*3/1000).*sin(2*pi*(1:1:nt)*dt*7/1000);

%%
k = min(size(zx)); %used to get the dimensionality of the approximant correctly.  Typically will be 1 unless you specify a k-dimensional target function.  

OMEGA =  G*(randn(N,N)).*(rand(N,N)<p)/(p*sqrt(N)); %Static weight matrix.  
z = zeros(k,1);  %initial approximant
BPhi = zeros(N,k); %initial decoder.  Best to keep it at 0.  
tspike = zeros(5*nt,2);  %If you want to store spike times, 
ns = 0; %count toal number of spikes
BIAS = 1000; %Bias current, note that the Rheobase is around 950 or something.  I forget the exact formula for this but you can test it out by shutting weights and feeding constant currents to neurons 
E = (2*rand(N,k)-1)*Q;  %Weight matrix is OMEGA0 + E*BPhi'; 
%% 
 Pinv = eye(N)*2; %initial correlation matrix, coefficient is the regularization constant as well 
 step = 20; %optimize with RLS only every 50 steps 
 imin = round(5000/dt); %time before starting RLS, gets the network to chaotic attractor 
 icrit = round(25000/dt); %end simulation at this time step 
 current = zeros(nt,k);  %store the approximant 
 RECB = zeros(nt,5); %store the decoders 
 REC = zeros(nt,10); %Store voltage and adaptation variables for plotting 
 i=1;
%% SIMULATION
tic
ilast = i ;
%icrit = ilast; %uncomment this, and restart cell if you want to test
% performance before icrit.  
for i = ilast:1:nt; 
%% EULER INTEGRATE
I = IPSC + E*z  + BIAS;  %postsynaptic current 
v = v + dt*(( ff.*(v-vr).*(v-vt) - u + I))/C ; % v(t) = v(t-1)+dt*v'(t-1)
u = u + dt*(a*(b*(v_-vr)-u)); %same with u, the v_ term makes it so that the integration of u uses v(t-1), instead of the updated v(t)
%% 
index = find(v>=vpeak);
if length(index)>0
JD = sum(OMEGA(:,index),2); %compute the increase in current due to spiking  
tspike(ns+1:ns+length(index),:) = [index,0*index+dt*i];  %uncomment this
%if you want to store spike times.  Takes longer.  
ns = ns + length(index); 
end

%synapse for single exponential 
if tr == 0 
    IPSC = IPSC*exp(-dt/td)+   JD*(length(index)>0)/(td);
    r = r *exp(-dt/td) + (v>=vpeak)/td;
else
    
%synapse for double exponential
IPSC = IPSC*exp(-dt/tr) + h*dt;
h = h*exp(-dt/td) + JD*(length(index)>0)/(tr*td);  %Integrate the current

r = r*exp(-dt/tr) + hr*dt; 
hr = hr*exp(-dt/td) + (v>=vpeak)/(tr*td);
end


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




%% Store, and plot.  
u = u + d*(v>=vpeak);  %implements set u to u+d if v>vpeak, component by component. 
v = v+(vreset-v).*(v>=vpeak); %implements v = c if v>vpeak add 0 if false, add c-v if true, v+c-v = c
v_ = v;  % sets v(t-1) = v for the next itteration of loop
REC(i,:) = [v(1:5)',u(1:5)'];  
current(i,:) = z';
RECB(i,:)=BPhi(1:5);


if mod(i,round(100/dt))==1 
dt*i
drawnow
gg = max(1,i - round(3000/dt));  %only plot for last 3 seconds
figure(2)
plot(dt*(gg:1:i)/1000,zx(:,gg:1:i),'k','LineWidth',2), hold on
plot(dt*(gg:1:i)/1000,current(gg:1:i,:),'b--','LineWidth',2), hold off
xlabel('Time (s)')
ylabel('$\hat{x}(t)$','Interpreter','LaTeX')
legend('Approximant','Target Signal')
xlim([dt*i-3000,dt*i]/1000)
figure(3)
plot((1:1:i)*dt/1000,RECB(1:1:i,:))
figure(14)
plot(tspike(1:ns,2),tspike(1:ns,1),'k.')
ylim([0,100])
end   

end
%%
tspike = tspike(tspike(:,2)~=0,:); 
M = tspike(tspike(:,2)>dt*icrit); 
AverageFiringRate = 1000*length(M)/(N*(T-dt*icrit))
%% Plot neurons before and after learning
figure(30)
for j = 1:1:5
plot((1:1:i)*dt/1000,REC(1:1:i,j)/(vpeak-vreset)+j), hold on 
end
xlim([T/1000-2,T/1000])
xlabel('Time (s)')
ylabel('Neuron Index') 
title('Post Learning')
figure(31)
for j = 1:1:5
plot((1:1:i)*dt/1000,REC(1:1:i,j)/(vpeak-vreset)+j), hold on 
end
xlim([0,imin*dt/1000])
xlabel('Time (s)')
ylabel('Neuron Index') 
title('Pre-Learning')
figure(40)
Z = eig(OMEGA+E*BPhi'); %eigenvalues after learning
Z2 = eig(OMEGA);  %eigenvalues before learning 
%% Plot eigenvalues before and after learning
plot(Z2,'r.'), hold on 
plot(Z,'k.') 
legend('Pre-Learning','Post-Learning')
xlabel('Re \lambda')
ylabel('Im \lambda')

