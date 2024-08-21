data = readtable("graph_data.csv");
x = data.x;
y = str2double(data.y);

p = polyfit(x,y,6);
%%
x_test = -50:100;

y_test = polyval(p,x_test);

plot(x_test,y_test,DisplayName='Polinomio fiteado')
hold on
plot(x,y,DisplayName='Data')
title('Linear Fit of Data')
legend;