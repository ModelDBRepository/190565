function dy = vanderpol(mu,MD,TC,t,y) 
dy(1,1) = mu*(y(1,1)-(y(1,1).^3)*(MD*MD/3) - y(2,1));
dy(2,1) = y(1,1)/mu; 
dy = dy*TC; 
end