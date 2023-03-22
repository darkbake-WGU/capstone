# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 18:12:03 2023
This program tests the random forest algorithm's effectiveness at predicting stock prices.
It uses Open, Close, High, Low, and Volume of the stock forecast_out days prior to predict the current price.
We split the data into training and test sets and scale the data.
Then we train a random forest algorithm on the data including using a grid search for the best parameters.
Then we run the test data through and graph the actual and predicted prices, and get the test metrics.
Notice that the data has been shuffled - this was necessary to train the random forest model. 
There is no way to unshuffle the data prior to making the graph.

https://towardsdatascience.com/predicting-the-price-of-the-beyond-meat-stock-using-random-forest-in-python-ebeae6aa9d49
@author: zebov
"""
#Imports
import yfinance as yf
import seaborn as sns
import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics

#Import data from yfinance
data = yf.Ticker('IDA')
df = data.history(period="max",  start="2017-01-01", end="2023-03-01")[['Open', 'Close', 'High', 'Low', 'Volume']]
print(df.head())

#Plot using Seaborn
sns.set()
df['timestamp'] = df.index
df['timestamp'] = pd.to_datetime(df['timestamp'])
sns.lineplot(df['timestamp'], df['Open'])
plt.ylabel("Open Price")

#Make prediction column
#Notice that we also modified the original code to use 5 features, not just 1.
forecast_out = 5
df['prediction'] = df[['Close']].shift(-forecast_out)
#X = np.array(df['Close']).reshape(-1,1)
X = df[['Open', 'Close', 'High', 'Low', 'Volume']].values
X = X[:-forecast_out]
y = np.array(df['prediction'])
y = y[:-forecast_out]

#Use test train split
#In this case, we are using the previous data from 3 days ago but this is shifted to be in the same time slot so random
#shuffle is necessary and will not hamper the results. I did test both, and turning off the shuffle broke the algorithm.
#Just leave the shuffle parameter out.
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.26,  random_state=0)

#Scale the features
scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)

#Hyper parameter tuning
model = RandomForestRegressor()
grid_rf = {
'n_estimators': [20, 50, 100, 500, 1000],  
'max_depth': np.arange(1, 15, 1),  
'min_samples_split': [2, 10, 9], 
'min_samples_leaf': np.arange(1, 15, 2, dtype=int),  
'bootstrap': [True, False], 
'random_state': [1, 2, 30, 42]
}

rscv = RandomizedSearchCV(estimator=model, param_distributions=grid_rf, cv=3, n_jobs=-1, verbose=2, n_iter=200)
rscv_fit = rscv.fit(x_train, y_train)
best_parameters = rscv_fit.best_params_
print(best_parameters)

#Apply model and predict
model = RandomForestRegressor(**best_parameters)
model.fit(x_train, y_train)
predict = model.predict(x_test)
print(predict.shape)

#Here we calculate the rmse and mape scores
rmse = metrics.mean_squared_error(y_test, predict, squared=False)
mape = metrics.mean_absolute_percentage_error(y_test, predict)

#Test with metrics
print('Random Forest Test Metrics')
print("Mean Absolute Error:", round(metrics.mean_absolute_error(y_test, predict), 4))
print("RMSE: ", rmse)
print("MSE: ", rmse*rmse)
print("MAPE: ", mape)
print("(R^2) Score:", round(metrics.r2_score(y_test, predict), 4))
print(f'Train Score : {model.score(x_train, y_train) * 100:.2f}% and Test Score : {model.score(x_test, y_test) * 100:.2f}% using Random Tree Regressor.')
errors = abs(predict - y_test)
#mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.') 


import matplotlib.pyplot as plt

# Create a list of dates for x-axis labels
dates = df.tail(len(y_test)).index

# Plot the predicted and actual stock prices
plt.clf()
plt.plot(dates, predict, label='Predicted Prices')
plt.plot(dates, y_test, label='Actual Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price (In Dollars)')
plt.legend()
plt.xticks(rotation=90)
plt.show()