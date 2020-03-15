# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:54:47 2019

@author: Richard_Fu
"""


import helper
import ml
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau



#df = pd.read_csv('test_data_switzerland.csv',delimiter=';')
#df = df.drop([0,1,2,3,4,5,6,7,8,9,10])
#df = df.drop(['Unnamed: 4'],axis=1)
#df = df.rename(columns={'LAT':'year','Unnamed: 1':'month','Unnamed: 2': 'day', 'Unnamed: 3':'hour'})
#df = df.rename(columns={'47.5584':'temp','47.5584.1':'cloud_cover','47.5584.2':'sun_duration','47.5584.3':'radiation'})
#df = df.reset_index(drop=True)
#df.temp = df.temp.astype(float)
#df.cloud_cover = df.cloud_cover.astype(float)
#df.sun_duration = df.sun_duration.astype(float)
#df.radiation = df.radiation.astype(float)
##df = df.drop(['year','month','day','hour'],axis=1)
#solar = helper.pv_data(address = 'Toronto,Canada',solar_size_kw_dc=50)
#solar = solar.reindex(index=np.roll(solar.index,1)).reset_index(drop=True)
#solar_prev = solar.reindex(index=np.roll(solar.index,1*24)).reset_index(drop=True)
#solar_2prev = solar.reindex(index=np.roll(solar.index,2*24)).reset_index(drop=True)
#solar_3prev = solar.reindex(index=np.roll(solar.index,3*24)).reset_index(drop=True)
#solar_4prev = solar.reindex(index=np.roll(solar.index,4*24)).reset_index(drop=True)
#solar_5prev = solar.reindex(index=np.roll(solar.index,5*24)).reset_index(drop=True)
#
#pred_rad = df.reindex(index=np.roll(df.index,-1*24)).reset_index(drop=True).radiation
#
#df['pred_rad'] = pred_rad
#df['solar_output_kw'] = solar.solar_output_kw
#df['solar_output_kw_prev'] = solar_prev.solar_output_kw
#df['solar_output_kw_2prev'] = solar_2prev.solar_output_kw
#df['solar_output_kw_3prev'] = solar_3prev.solar_output_kw
#df['solar_output_kw_4prev'] = solar_4prev.solar_output_kw
#df['solar_output_kw_5prev'] = solar_5prev.solar_output_kw
#df = df.drop(columns=['year','month','day','hour','sun_duration'])


#test_df = helper.load_data('anika_therapeutics.p')
#test_df = test_df.dropna()
#solar = helper.pv_data(address='Bedford,Massachusetts',solar_size_kw_dc=144,tilt=20)
#test_df = test_df[0:105000]
#df = helper.add_solar_to_df(test_df,solar,hourly_intervals=12)
#df = df.drop(columns=['date'])


df = pd.read_csv('new_york_weather_year.csv')
df.drop_duplicates(subset="dt_iso",keep=False,inplace=True)
df.dt_iso = pd.to_datetime(df.dt_iso, format='%Y-%m-%d %H:%M:%S +0000 UTC')
df = df.drop(columns=['dt','city_id','city_name','lat','lon','sea_level','grnd_level','rain_1h','rain_3h','rain_24h',
                 'snow_1h','snow_3h','snow_24h','snow_today','weather_main','weather_description','weather_icon','rain_today'])
df = df.dropna()
df = df.rename(columns={'dt_iso':'date'})


solar = helper.pv_data(address='New York,NY', solar_size_kw_dc = 50, tilt=30)
with_solar = helper.add_solar_to_df(df,solar,1)
df = with_solar
df = df.drop(columns=['date'])

target_output = ['solar_output_kw','temp']
shift_days = 1
shift_steps = shift_days*24
target_df = df[target_output]

x = df.drop(columns=target_output).values
y = target_df.values
training_num = int(0.8*len(x))

x_train = x[0:training_num]
x_test = x[training_num:]

y_train = y[0:training_num]
y_test = y[training_num:]

#scaling the data (sigmoid that shit) (quite important if the range of data is very high)

x_scaler = MinMaxScaler()
x_train_scaled = x_scaler.fit_transform(x_train)
x_test_scaled = x_scaler.fit_transform(x_test)

y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.fit_transform(y_test)

#batch testing? - not sure if needed. definately needed for larger datasets
#batch size dependent on gpu power
#sequence length is the number of observations for each 

batch_size = 256
sequence_length = 24

batch = ml.batch_generator(batch_size = batch_size,sequence_length=sequence_length,x_data=x,y_data=y,
                                            training_number=training_num,x_train_scaled=x_train_scaled,y_train_scaled=
                                            y_train_scaled)
x_batch,y_batch = next(batch)
print(x_batch.shape)
print(y_batch.shape)

#batch = 0
#signal = 0
#seq = x_batch[batch,:,signal]
#plt.plot(seq)

#validation set

validation_data = (np.expand_dims(x_test_scaled, axis=0),
                   np.expand_dims(y_test_scaled, axis=0))


#building the recurrent neural network
model = Sequential()
#adding a gated recurrent unit to this model for memory (similar to lstn except a bit easier to understand)

model.add(LSTM(units=64,return_sequences=True,input_shape=(None,x.shape[1],))) 
model.add(Dense(64,activation = 'tanh'))    

#mapping down to 2 output units from the 512 hidden layer units using sigmoid function

model.add(Dense(y.shape[1],activation='tanh'))

#alternatively, using a linear activation function may be more accurate, however, it is not necessary within context 

#loss function generation- mean squared error 

#warmup period - this is inaccurate data bc the thing just began 

#optimizer = adam(lr=1e-3)
model.compile(loss='mae',optimizer='adam',metrics=['accuracy'])


#callbacks are for logging process...just copy pasted this stuff 

path_checkpoint = '23_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)

callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=5, verbose=1)

callback_tensorboard = TensorBoard(log_dir='./23_logs/',
                                   histogram_freq=0,
                                   write_graph=False)

callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-4,
                                       patience=0,
                                       verbose=1)

callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard,
             callback_reduce_lr]


model.fit_generator(generator=batch,
                    epochs=20,
                    steps_per_epoch=100,
                    validation_data=validation_data,
                    callbacks=callbacks)

result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
                        y=np.expand_dims(y_test_scaled, axis=0))

start_idx = 0
length = 1000
target_names = target_output
y_pred, y_true,x = ml.plot_comparison(x_test_scaled,y_test,y_scaler,
                                                       target_names,output)

