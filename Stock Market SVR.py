# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 17:34:51 2023
This program tests the SVM algorithm's effectiveness at predicting stock prices.'
@author: zebov
"""

# Import Python Libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import datetime as dt

#Import yfinance
import yfinance as yf

stock='IDA'

# Download stock data for the stock you want using the ticker and start and end dates, and the interval.
data = yf.download(stock, start='2017-01-01', end='2023-03-01', interval='1d')

# Set the index of the dataframe to the 'Date' column
data = data.reset_index(drop=True)

#Set the data to only use the adj close column
data = data[['Open', 'Close', 'High', 'Low', 'Adj Close']]

#Notice that the date column is automatically the index
print(data.head())

#Create one more column Prediction shifted n days up. 
n=5
data['Prediction'] = data['Adj Close'].shift(-n)

# Drop the last 15 rows of the dataset, as these do not have 'Predictions'
data = data[:-n]


#Create a data set X and convert it into numpy array , which will be having actual values
X = np.array(data.drop(['Prediction'],1))
print(X)

# Create a dataset y which will be having Predicted values and convert into numpy array
y = np.array(data['Prediction'])

# Split the data into train and test with 90 & 10 % respectively
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

# SVM Model
svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
# Train the model 
svr.fit(x_train, y_train)

# The best possible score is 1.0
svm_confidence = svr.score(x_test, y_test)
print("svm confidence: ", svm_confidence)

y_pred = svr.predict(x_test.reshape(-1,5))

import matplotlib.pyplot as plt

# Plot actual vs predicted prices
plt.plot(y_test, label='Actual Prices')
plt.plot(y_pred, label='Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Observation')
plt.ylabel('Price')
plt.legend()
plt.show()

#Here we calculate the rmse and mape scores
rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
mape = metrics.mean_absolute_percentage_error(y_test, y_pred)

#Test with metrics
print('SVR Test Metrics:')
print("Mean Absolute Error:", round(metrics.mean_absolute_error(y_test, y_pred), 4))
print("RMSE: ", rmse)
print("MSE: ", rmse*rmse)
print("MAPE: ", mape)
print("(R^2) Score:", round(metrics.r2_score(y_test, y_pred), 4))
errors = abs(y_pred - y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')




forecast = np.array(data.drop(['Prediction'],1))[-15:]
print(forecast)

# support vector model predictions for the next ‘15’ days
svm_prediction = svr.predict(forecast)
print(svm_prediction)