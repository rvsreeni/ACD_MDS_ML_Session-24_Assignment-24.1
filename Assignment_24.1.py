#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 00:12:33 2018

@author: macuser
"""
#import datetime
import pandas as pd
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas.tools.plotting import autocorrelation_plot
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')

series = pd.read_csv('shampoo-sales.csv', header=0, parse_dates=[0], 
                  index_col=0, date_parser=parser)
    
#print(series.shape)
#autocorrelation_plot(series)
#plt.show()

# fit model
#model = ARIMA(series, order=(5,1,0))
#model_fit = model.fit(disp=0)
#print(model_fit.summary())

# plot residual errors
#residuals = DataFrame(model_fit.resid)
#residuals.plot()
#plt.show()
#residuals.plot(kind='kde')
#plt.show()
#print(residuals.describe())


X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()