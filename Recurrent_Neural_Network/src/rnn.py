# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 12:39:44 2019

@author: Ahmed Alhag
"""

# Recurrent Neural Network

# Check that the gpu/cpu is being used properly
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Import Default Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Import datasets
BinPath = os.path.abspath(os.path.join(os.path.dirname( '__file__' ), '..', 'bin'))
TrainingData = BinPath + '/Google_Stock_Price_Train.csv';
TestingData = BinPath + '/Google_Stock_Price_Test.csv';

dataset_train = pd.read_csv(TrainingData)
dataset_test = pd.read_csv(TestingData)

training_set = dataset_train.iloc[:, 1:2].values
real_stock_price = dataset_test.iloc[:, 1:2].values
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

# Data structure with N timesteps
N = 60
X_train = []
Y_train = []
for i in range(N, 1258):
    X_train.append(training_set_scaled[i-N:i, 0])
    Y_train.append(training_set_scaled[i, 0])
X_train, Y_train = np.array(X_train), np.array(Y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Initialising the RNN
regressor = Sequential()

# First LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Second LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Third LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Fourth LSTM layer
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer='adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, Y_train, epochs = 100, batch_size = 32)

# Getting the predicted stock price of 2017
inputs = dataset_total[len(dataset_total) - len(dataset_test) - N:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(N, 80):
    X_test.append(inputs[i-N:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()