# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 17:44:23 2017

@author: Bahar Dorri

"""
import numpy as np
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot as plt

def dateparse (time_in_secs):    
    return datetime.fromtimestamp(float(time_in_secs))

#data should divide into input (X) and output (y) components
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

# create a differenced series-stationary-remove trend
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    
    return yhat + history[-interval]

 # scale train and test data to [-1, 1]
def scale(train, test):
	 # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1]) #matrix format
    train_scaled = scaler.transform(train)
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

    # transform test
# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]
 
def fit_lstm(train, batch_size,nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    #The LSTM layer expects input to be in a matrix with the dimensions: [samples, time steps, features].
    X = X.reshape(X.shape[0], 1, X.shape[1]) 
        
    model = Sequential() #make linear stack of layers
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True, return_sequences=True))
    model.add(Dropout(0.3)) 
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dropout(0.2))  
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):    
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    
    return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]

series = read_csv('C:/Users/mohse/Desktop/bahar classes/Fall2017/deep learning/project/bitcoin-historical-data/krakenEUR_1-min_data_2014-01-08_to_2017-05-31.csv', parse_dates=True, date_parser=dateparse, index_col=[0])
series['Close'].fillna(method='ffill', inplace=True)
totalData=series['Close']
NewData=list()
#for 300 days between 1000000 min and 1440000min of data
for j in range(0,1700000,1440):
       NewData.append(totalData[j])
print('NewData',len(NewData))

raw_Part = NewData
differenced = difference(raw_Part, 1)
supervised = timeseries_to_supervised(differenced, 1)
supervised_values = supervised.values

train, test = supervised_values[700:1000], supervised_values[1000:1100]
raw_test=raw_Part[1000:1100]
print('raw',len(raw_test))

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)
print('len(test_scaled)',len(test_scaled))
#print('scale',train_scaled)
# fit the model
batch_size=1 #This is because it must be a factor of the size of the training and test datasets.
repeats=1
error_scores=list()
for r in range(repeats):
    lstm_model = fit_lstm(train_scaled, batch_size,500, 50)

# forecast the entire training dataset to build up state for forecasting
    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)#requires a 3D NumPy array as input 
    lstm_model.predict(train_reshaped, batch_size=batch_size)
# walk-forward validation on the test data
    predictions = list()
    for i in range(len(test_scaled)):
        # make one-step forecast
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        yhat = forecast_lstm(lstm_model, batch_size, X)
        yhat = invert_scale(scaler, X, yhat)
        yhat = inverse_difference(raw_test, yhat, len(test_scaled)-i)
        # store forecast
        predictions.append(yhat)
        expected = raw_test[i]
        print('Predicted=%f, Expected=%f' % (yhat, round(expected,3)))
        
    
    rmse = sqrt(mean_squared_error(raw_test, predictions))
    print('%d) Test RMSE: %.3f' % (r+1, rmse))
    error_scores.append(rmse)

plt.plot(raw_test, 'orange', label='expected')
plt.plot(predictions,'blue', label='predictions')
plt.title('predictions and expected')
plt.legend()
plt.show()

t=totalData[1700000]
t=np.array(t)
t = t.reshape(1, 1, 1)
yhat2 = lstm_model.predict(t, batch_size=batch_size) 
yhat2 = invert_scale(scaler, t, yhat2)
yhat2= inverse_difference(t, yhat2)  
print('Predicted=%f, Expected=%f' % (yhat2, totalData[1701440]))



