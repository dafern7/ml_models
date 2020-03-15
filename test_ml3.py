# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:01:42 2019

@author: Richard_Fu
"""

from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM


import helper
import useful_ml_functions
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os


import helper
import numpy as np
solar = helper.pv_data(address = 'New York City, NY', solar_size_kw_dc = 25, tilt = 20)

import pandas as pd
data = pd.read_csv('nyc_data2017.csv')
data.columns = ['year','month','day','hour','minute','dhi','dni','ghi','x','surface_albedo','xx','pressure']
data = data.drop([0,1])
data = data.reset_index(drop=True)
data = data.drop(columns=['x','xx','minute'])
data['solar_output_kw'] = solar.solar_output_kw

solar_prev = solar.reindex(index=np.roll(solar.index,1*24)).reset_index(drop=True)
solar_2prev = solar.reindex(index=np.roll(solar.index,2*24)).reset_index(drop=True)
dni_prev = data.reindex(index=np.roll(data.index,1*24)).reset_index(drop=True).dni
dni_2prev = data.reindex(index=np.roll(data.index,2*24)).reset_index(drop=True).dni
dhi_prev = data.reindex(index=np.roll(data.index,1*24)).reset_index(drop=True).dhi
dhi_2prev = data.reindex(index=np.roll(data.index,2*24)).reset_index(drop=True).dhi
ghi_prev = data.reindex(index=np.roll(data.index,1*24)).reset_index(drop=True).ghi
ghi_2prev = data.reindex(index=np.roll(data.index,2*24)).reset_index(drop=True).ghi

pred_dni = data.reindex(index=np.roll(data.index,-1*24)).reset_index(drop=True).dni
pred_dhi = data.reindex(index=np.roll(data.index,-1*24)).reset_index(drop=True).dhi
pred_ghi = data.reindex(index=np.roll(data.index,-1*24)).reset_index(drop=True).ghi

pred_dni2 = data.reindex(index=np.roll(data.index,-2*24)).reset_index(drop=True).dni
pred_dhi2 = data.reindex(index=np.roll(data.index,-2*24)).reset_index(drop=True).dhi
pred_ghi2 = data.reindex(index=np.roll(data.index,-2*24)).reset_index(drop=True).ghi



data['predicted_dni'] = pred_dni.astype(float)
data['predicted_dhi'] = pred_dhi.astype(float)
data['predicted_ghi'] = pred_ghi.astype(float)


#data['predicted_dni2'] = pred_dni2.astype(float)
#data['predicted_dhi2'] = pred_dhi2.astype(float)
#data['predicted_ghi2'] = pred_ghi2.astype(float)



#data['dni_prev'] = dni_prev
#data['dni_2prev'] = dni_2prev
#data['dhi_prev'] = dhi_prev
#data['dhi_2prev'] = dhi_2prev
#data['ghi_prev'] = ghi_prev
#data['ghi_2prev'] = ghi_2prev
data['solar_output_kw'] = solar.solar_output_kw.astype(float)
#data['solar_output_kw_prev'] = solar_prev.solar_output_kw
#data['solar_output_kw_2prev'] = solar_2prev.solar_output_kw

data = data.drop(columns=['year','month','day','hour'])#,'dni','dhi','ghi','surface_albedo','pressure'])


#other_data = useful_ml_functions.call_darksky(40.77,-73.98,2017)

#data['pred_temperature'] = other_data.reindex(index=np.roll(other_data.index,1*24)).reset_index(drop=True).temp
#data['pred_humidity'] = other_data.reindex(index=np.roll(other_data.index,1*24)).reset_index(drop=True).humidity
#data['pred_cloud_cover'] = other_data.reindex(index=np.roll(other_data.index,1*24)).reset_index(drop=True).cloud_cover
#data['pred_uv'] = other_data.reindex(index=np.roll(other_data.index,1*24)).reset_index(drop=True).uv_index
#data['pred_precip_prob'] = other_data.reindex(index=np.roll(other_data.index,1*24)).reset_index(drop=True).precipitation_probability



#data.pred_temperature = data.pred_temperature.astype(float)
#data.pred_humidity = data.pred_humidity.astype(float)
#data.pred_cloud_cover = data.pred_cloud_cover.astype(float)
#data.loc[(data.predicted_ghi == 0),'pred_cloud_cover'] = 0
#data.pred_uv = data.pred_uv.astype(float)
#data.pred_precip_prob = data.pred_precip_prob.astype(float)
data.dni = data.dni.astype(float)
data.dhi = data.dhi.astype(float)
data.ghi = data.ghi.astype(float)
data.surface_albedo = data.surface_albedo.astype(float)
data.pressure = data.pressure.astype(float)
df = data


def split_dataset(data):
	# split into standard weeks
	train, test = data[0:-1440], data[-1441:-1]
	# restructure into windows of weekly data
	train = array(split(train, len(train)/24))
	test = array(split(test, len(test)/24))
	return train, test


train,test = split_dataset(df.values)

print(train.shape)
print(train[0, 0, 0], train[-1, -1, 0])
# validate test
print(test.shape)
print(test[0, 0, 0], test[-1, -1, 0])

new = train.reshape((train.shape[0]*train.shape[1],train.shape[2]))


def evaluate_forecasts(actual, predicted):
    scores = list()
    score = list()
    pred = list()
    act = list()
    
    for i in range(actual.shape[1]):
        mse = mean_squared_error(actual[:,i], predicted[:,i])
        rmse = sqrt(mse)
        scores.append(rmse)
    
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            score.append(actual[row,col]-predicted[row,col])
            act.append(actual[row,col])
            pred.append(predicted[row,col])
    return score,scores,pred,act
            
    

def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))
    
#splitting dataset
def to_supervised(train, n_input, n_out=24):
	# flatten data
	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	X, y = list(), list()
	in_start = 0
	# step over the entire history one time step at a time
	for _ in range(len(data)):
		# define the end of the input sequence
		in_end = in_start + n_input
		out_end = in_end + n_out
		# ensure we have enough data for this instance
		if out_end <= len(data):
			x_input = data[in_start:in_end, 0]
			x_input = x_input.reshape((len(x_input), 1))
			X.append(x_input)
			y.append(data[in_end:out_end, 0])
		# move along one time step
		in_start += 1
	return array(X), array(y)

def build_model(train, n_input):
	# prepare data
	train_x, train_y = to_supervised(train, n_input)
	# define parameters
	verbose, epochs, batch_size = 0, 25, 128
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# define model
	model = Sequential()
	model.add(LSTM(512, activation='relu', input_shape=(n_timesteps, n_features)))
	model.add(Dense(512, activation='relu'))
	model.add(Dense(n_outputs))
	model.compile(loss='mse', optimizer='adam')
	# fit network
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model

def forecast(model, history, n_input):
	# flatten data
	data = array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	# retrieve last observations for input data
	input_x = data[-n_input:, 0]
	# reshape into [1, n_input, 1]
	input_x = input_x.reshape((1, len(input_x), 1))
	# forecast the next week
	yhat = model.predict(input_x, verbose=0)
	# we only want the vector forecast
	yhat = yhat[0]
	return yhat


def evaluate_model(train, test, n_input):
	# fit model
	model = build_model(train, n_input)
	# history is a list of weekly data
	history = [x for x in train]
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = forecast(model, history, n_input)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	# evaluate predictions days for each week
	predictions = array(predictions)
	score, scores,pred,act = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores, pred, act

train, test = split_dataset(df.values)
# evaluate model and get scores
n_input = 14
score, scores, pred, act = evaluate_model(train, test, n_input)
# summarize scores

# plot scores
pyplot.plot(pred, label='predicted')
pyplot.plot(act,label='actual')
pyplot.show()




    
    