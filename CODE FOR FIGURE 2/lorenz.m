function dy = lorenz(TC,sigma,rho,beta,MD,t,y) 

dy(1,1) = sigma*(y(2,1) - y(1,1)); 
dy(2,1) = rho*y(1,1)-MD*y(1,1)*y(3,1)-y(2,1);
dy(3,1) = MD*y(1,1)*y(2,1)-beta*y(3,1);
dy = TC*dy;
end