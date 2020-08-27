# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 23:59:17 2020

@author: himan
"""

import sys
sys.path.append('/content/drive/My Drive/Colab Notebooks/TensorFlow 2.0/modules')
import pandas as pd
import tf_dataset_extractor as e
#import grapher_v1_1 as g
#import LSTM_creator_v1_0 as l
v = e.v
g = e.g
l = e.l

from google.colab import drive
drive.mount('/content/drive')

# load dataset
v.upload.online_csv('/content/drive/My Drive/Colab Notebooks/TensorFlow 2.0/csv/GOOG.csv')
e.K = v.upload.make_backup()

#original copy without preprocessing
v.upload.retrieve_backup(e.K)

#dropping extra columns
e.X = e.X.drop(['High', 'Low', 'Close', 'Adj Close', 'Volume'], axis=1)

#preprocessing
index = e.X.pop('Date')
scaler, e.X = v.partition.scale('all_df', scaler='MinMaxScaler', df=e.X, to_float=True, return_df=True)
e.X = e.X.set_index(index)
e.X = l.preprocessing.series_to_supervised(e.X, 3, 1)

#X, y
v.extract.labels(['var1(t)'])

#train, test
X_train_, X_test_ = l.preprocessing.split(0.1, e.X)
y_train_, y_test_ = l.preprocessing.split(0.1, e.y)
e.X_ = e.X.copy()
e.y_ = e.y.copy()
print(X_train_.shape, X_test_.shape, y_train_.shape, y_test_.shape)

import matplotlib.pyplot as plt

fig=plt.figure(figsize=(20, 10), dpi= 80)
fig=plt.plot(e.y_)

#normal preprocessing
v.upload.retrieve_backup(e.K)

#dropping extra columns
e.X = e.X.drop(['High', 'Low', 'Close', 'Adj Close', 'Volume'], axis=1)

#preprocessing
index = e.X.pop('Date')
scaler, e.X = v.partition.scale('all_df', scaler='MinMaxScaler', df=e.X, to_float=True, return_df=True)
e.X = e.X.set_index(index)
l.preprocessing.transform_to_stationary()
e.X = l.preprocessing.series_to_supervised(e.X, 3, 1, drop_col=False)

#X, y
v.extract.labels(['var1(t)'])

#train, test
X_train, X_test = l.preprocessing.split(0.1, e.X)
y_train, y_test = l.preprocessing.split(0.1, e.y) #sembra non servire a nulla
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

import matplotlib.pyplot as plt

fig=plt.figure(figsize=(20, 10), dpi= 80)
fig=plt.plot(e.y)

import matplotlib.pyplot as plt

fig=plt.figure(figsize=(20, 10), dpi= 80)
fig=plt.plot(e.X)

#reshape [samples, n_input_timesteps, n_features]
X_train = X_train.reshape((225, 3, 1))
y_train = y_train.reshape((225, 1, 1))
print(X_train.shape, y_train.shape)
#every sample has dimensions [1, 3, 1]

# Commented out IPython magic to ensure Python compatibility.
#LSTM
# %tensorflow_version 2.x
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

model = Sequential()
model.add(LSTM(50, batch_input_shape=(1, 3, 1), stateful=True)) #dimensions of every single sample
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, epochs=3000, batch_size=1, verbose=2, shuffle=False)
model.reset_states()

X_test = X_test.reshape(24, 3, 1)
y_test = y_test.reshape(24, 1, 1)
print(X_test.shape, y_test.shape)

#make a one-step forecast
yhat = model.predict(X_test, verbose=2, batch_size=1)
#without batch_size the model only accepts one input at a time
yhat

#invert preprocessing on predicted data
#remove stationary
y_test = y_test.reshape(24, 1)
var1 = y_test_    #original values
var2 = yhat       #gaps
var3 = list()     #

#var1 = var1.values
#var2 = var2.values

var3.append(var1[0])
for i in range(0, len(var2)):
  values = var1[i] + var2[i]
  var3.append(values)
var3

#inverse scaling
predicted = scaler.inverse_transform(var3)
predicted

#invert preprocessing on expected data
#inverse scaling
expected = scaler.inverse_transform(y_test_)
expected

for i in range(len(y_test_)):
  print('iteration=%d, Predicted=%f, Expected=%f' % (i+1, predicted[i], expected[i]))

import matplotlib.pyplot as plt

fig=plt.figure(figsize=(20, 10), dpi= 80)
fig=plt.plot(expected)
fig=plt.plot(predicted)

# report performance
from math import *
from sklearn.metrics import mean_squared_error
rmse = sqrt(mean_squared_error(expected, predicted))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted