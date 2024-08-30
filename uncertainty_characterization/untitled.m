data = readtable("total_eta_values.csv");
data = data(2:end,1);
data = table2array(data);
%%
distributionFitter