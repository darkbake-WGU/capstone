# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 17:20:12 2023

This program will test the effectiveness of an advanced LSTM model at predicting stock prices.
It uses the yfinance API to gather stock market data and outputs a graph of the predicted vs actual values
as well as various statistical measures on the effectiveness of the LSTM algorithm at predicting the stock over the test set.

This program uses ensemble learning, with n_models averaged out, to test the effectiveness of LSTM on predicting stock
prices. However, ensemble learning is set to n_models = 1 by default and not used in test metrics.

It does not predict future stock prices. Another program does that. This program is used to test parameters and investigate
the effectiveness of different settings on predicting stock prices.

Debugging:
    Notice that you can use intervals of 1d, 5d, 1mo for example in yfinance. Using longer intervals, ensemble learning messes
    with them, so use n_models = 1.

@author: zebov

"""
#Write a function to create windows of n steps, this is the fundamental LSTM Split Function for creating a time series
def lstm_split(data, n_steps):
    X, y = [], []
    for i in range(len(data)-n_steps+1):
        X.append(data[i:i + n_steps, :-1])
        y.append(data[i + n_steps-1, -1])

    return np.array(X), np.array(y)

#Import packages
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

from sklearn.preprocessing import StandardScaler
from sklearn import metrics

#Import yfinance
import yfinance as yf

# Download stock data for the stock you want using the ticker and start and end dates, and the interval.
data = yf.download('IDA', start='2017-01-01', end='2023-03-01', interval='1d')

#Notice that the date column is automatically the index
print(data.head())

#Plot the data for IDACORP

#Reset the index so that Date is a regular column
data2 = data.reset_index()

# Set the axis to show the correct dates
plt.figure(figsize=(15,10))
plt.plot(data2['Date'], data2['High'], label = 'High')
plt.plot(data2['Date'], data2['Low'], label = 'Low')
plt.xlabel('Date')
plt.ylabel('USD')
plt.legend()
plt.show()

# Define the target and the features
target_y = data['Close']
features_x = data.iloc[:,0:3]

print(features_x.head())

#Check the features header
print(features_x.head())

# Use feature scaling on the features and target variable separately
scaler_x = StandardScaler()
scaled_features = pd.DataFrame(scaler_x.fit_transform(features_x), columns=features_x.columns, index=features_x.index)

scaler_y = StandardScaler()
scaled_target = pd.Series(scaler_y.fit_transform(target_y.values.reshape(-1, 1)).squeeze(), index=target_y.index, name=target_y.name)

# Combine the scaled features and target into one DataFrame
scaled_data = pd.concat([scaled_features, scaled_target], axis=1)

#This calls the lstm_split function that makes windows of the past n_steps days predicting the next day.
#n_steps defines the number of days to look back for each prediction point
#NOTICE: Using less n_steps can actually be MORE accurate.
X1, y1 = lstm_split(scaled_data.values, n_steps=15)

#Check their shapes here. Things are going okay.
print(X1.shape)
print(y1.shape)

#Import train_test_split
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
#NOTICE set shuffle=False, or the graph at the end of this project will not be in temporal order.
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42, shuffle=False)

#Set up the LSTM Model
def build_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train and save multiple models for ensemble learning
#Turn off ensemble learning by setting n_models = 1
n_models = 1
models = []
for i in range(n_models):
    model = build_model()
    model.fit(X_train, y_train, batch_size=1, epochs=15)
    model.save(f"model_{i}.h5")
    models.append(model)

#We are going to use ensemble learning here
from keras.models import load_model

# Load the saved models
models = []
for i in range(n_models):
    model = load_model(f"model_{i}.h5")
    models.append(model)

# Make predictions with each model
y_preds = []
for model in models:
    y_pred = model.predict(X_test)
    y_preds.append(y_pred)

# Combine predictions by averaging them
y_ensemble = pd.DataFrame(np.mean(y_preds, axis=0), columns=['target'])

# Inverse transform the scaled target variable
# Unscale y_ensemble
y_ensemble = scaler_y.inverse_transform(y_ensemble)

# Reshape y_test
y_test = y_test.reshape(-1, 1)

#Inverse transform y_test
y_test = scaler_y.inverse_transform(y_test)

# Reshape y_test back to original shape
y_test = y_test.reshape(-1)

#Here we print the shapes of the predictions for troubleshooting
print('x_test:', X_test.shape)
print('y_pred:', y_ensemble.shape)
print('y_test:', y_test.shape)

#You can comment these back in to print a table of the results
#results = pd.DataFrame(data={'Predicted':y_pred, 'acutal': y_test})

#Here we have some code to graph the prediction vs. actual price
#Notice that the graph only shows the past 
import matplotlib.pyplot as plt

plt.plot(y_ensemble, label='Prediction')
plt.plot(y_test, label='Actual Price')
plt.xlabel('Date')
plt.ylabel('IDACORP Stock Price')
plt.legend()
plt.show()

#Here we calculate the rmse and mape scores
rmse = metrics.mean_squared_error(y_test, y_ensemble, squared=False)
mape = metrics.mean_absolute_percentage_error(y_test, y_ensemble)

#Test with metrics
print("LSTM Metrics:")
print("Mean Absolute Error:", round(metrics.mean_absolute_error(y_test, y_ensemble), 4))
print("RMSE: ", rmse)
print("MSE: ", rmse*rmse)
print("MAPE: ", mape)
print("(R^2) Score:", round(metrics.r2_score(y_test, y_ensemble), 4))
errors = abs(y_ensemble - y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.') 
