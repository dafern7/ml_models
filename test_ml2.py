# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:06:10 2019

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
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding, LSTM
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import keras

import helper
import numpy as np
solar = helper.pv_data(address = 'New York, NY', solar_size_kw_dc = 200, tilt = 20)

import pandas as pd
data = pd.read_csv('nyc_data2017.csv')
data.columns = ['year','month','day','hour','minute','dhi','dni','ghi','x','surface_albedo','xx','pressure']
data = data.drop([0,1])
data = data.reset_index(drop=True)
data = data.drop(columns=['x','xx','minute','year','month','day','hour'])
data['solar_output_kw'] = solar.solar_output_kw
#data['solar_irradiance'] = solar.solar_irradiance

#solar_prev = solar.reindex(index=np.roll(solar.index,1*24)).reset_index(drop=True)
#solar_2prev = solar.reindex(index=np.roll(solar.index,2*24)).reset_index(drop=True)
#dni_prev = data.reindex(index=np.roll(data.index,1*24)).reset_index(drop=True).dni
#dni_2prev = data.reindex(index=np.roll(data.index,2*24)).reset_index(drop=True).dni
#dhi_prev = data.reindex(index=np.roll(data.index,1*24)).reset_index(drop=True).dhi
#dhi_2prev = data.reindex(index=np.roll(data.index,2*24)).reset_index(drop=True).dhi
#ghi_prev = data.reindex(index=np.roll(data.index,1*24)).reset_index(drop=True).ghi
#ghi_2prev = data.reindex(index=np.roll(data.index,2*24)).reset_index(drop=True).ghi
#
#pred_dni = data.reindex(index=np.roll(data.index,-1*24)).reset_index(drop=True).dni
#pred_dhi = data.reindex(index=np.roll(data.index,-1*24)).reset_index(drop=True).dhi
#pred_ghi = data.reindex(index=np.roll(data.index,-1*24)).reset_index(drop=True).ghi
#
#pred_dni2 = data.reindex(index=np.roll(data.index,-2*24)).reset_index(drop=True).dni
#pred_dhi2 = data.reindex(index=np.roll(data.index,-2*24)).reset_index(drop=True).dhi
#pred_ghi2 = data.reindex(index=np.roll(data.index,-2*24)).reset_index(drop=True).ghi
#
#rad_pred = solar.reindex(index=np.roll(solar.index,1*24)).reset_index(drop=True).solar_irradiance
#data['previous_irradiation'] = rad_pred.astype(float)
#data['predicted_dni'] = pred_dni.astype(float)
#data['predicted_dhi'] = pred_dhi.astype(float)
#data['predicted_ghi'] = pred_ghi.astype(float)


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
data['solar_irradiance'] = solar.solar_irradiance.astype(float)/1000
#data['solar_output_kw_prev'] = solar_prev.solar_output_kw
#data['solar_output_kw_2prev'] = solar_2prev.solar_output_kw

#data = data.drop(columns=['year','month','day','hour','dhi','ghi','surface_albedo','pressure'])


#other_data = ml.call_darksky(40.77,-73.98,2017)

#
##data['pred_uv'] = other_data.reindex(index=np.roll(other_data.index,1*24)).reset_index(drop=True).uv_indexdata['pred_temperature'] = other_data.reindex(index=np.roll(other_data.index,0*24)).reset_index(drop=True).temp
#data['pred_humidity'] = other_data.reindex(index=np.roll(other_data.index,0*24)).reset_index(drop=True).humidity
#data['pred_cloud_cover'] = other_data.reindex(index=np.roll(other_data.index,0*24)).reset_index(drop=True).cloud_cover
##data['pred_precip_prob'] = other_data.reindex(index=np.roll(other_data.index,1*24)).reset_index(drop=True).precipitation_probability
#
#
#
#data.pred_temperature = data.pred_temperature.astype(float)
#data.pred_humidity = data.pred_humidity.astype(float)
#data.pred_cloud_cover = data.pred_cloud_cover.astype(float)
#data.loc[(data.solar_irradiance== 0),'pred_cloud_cover'] = 0
#data.pred_uv = data.pred_uv.astype(float)
#data.pred_precip_prob = data.pred_precip_prob.astype(float)
#data.dni = data.dni.astype(float)
#data.dhi = data.dhi.astype(float)
#data.ghi = data.ghi.astype(float)
#data.surface_albedo = data.surface_albedo.astype(float)
#data.pressure = data.pressure.astype(float)

        

from sklearn.preprocessing import MinMaxScaler
target_output = ['solar_irradiance','solar_output_kw']
shift_days = 1
shift_steps = shift_days*24
target_df = data[target_output]

x = data.drop(columns=target_output).values
y = target_df.values
training_num = int(0.7*len(x))
val_num = int(0.1*len(x))

x_train = x[0:training_num]
x_val = x[training_num:training_num+val_num]
x_test = x[training_num:]

y_train = y[0:training_num]
y_val = y[training_num:training_num+val_num]
y_test = y[training_num:]

x_scaler = MinMaxScaler()
x_train_scaled = x_scaler.fit_transform(x_train)
x_val_scaled = x_scaler.fit_transform(x_val)
x_test_scaled = x_scaler.fit_transform(x_test)

y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_val_scaled = y_scaler.fit_transform(y_val)
y_test_scaled = y_scaler.fit_transform(y_test)

batch_size = 256
sequence_length = 24

batch = ml.batch_generator(batch_size = batch_size,sequence_length=sequence_length,x_data=x,y_data=y,
                                            training_number=training_num,x_train_scaled=x_train_scaled,y_train_scaled=
                                            y_train_scaled)

#iterate through the batchs
#batch is a generator object
x_batch,y_batch = next(batch)
print(x_batch.shape)
print(y_batch.shape)

#need to make the data set 3 dimensional tensors
validation_data = (np.expand_dims(x_val_scaled, axis=0),
                   np.expand_dims(y_val_scaled, axis=0))




#building the recurrent neural network
model = Sequential()
#model.reset_states()
#adding a gated recurrent unit to this model for memory (similar to lstn except a bit easier to understand)

#model.add(LSTM(units=128, return_sequences=True, input_shape=(None,x.shape[1],)))

#model.add(LSTM(units=128, return_sequences=True, input_shape=(None,256,)))

#model.add(GRU(units=64, return_sequences=True, input_shape=(None,x.shape[1],)))

#model.add(Dense(256,activation='relu'))
model.add(LSTM(units=512,return_sequences=True,input_shape=(None,x.shape[1],)))
#model.add(Dense(1024,activation='relu'))

model.add(Dense(64,activation = 'tanh'))


#mapping down to 2 output units from the 512 hidden layer units using sigmoid function

model.add(Dense(y.shape[1],activation='tanh'))

#alternatively, using a linear activation function may be more accurate, however, it is not necessary within context 

#loss function generation- mean squared error 


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
                    epochs=15,
                    steps_per_epoch=100,
                    validation_data=validation_data,
                    callbacks=callbacks)

result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
                        y=np.expand_dims(y_test_scaled, axis=0))

start_idx = 0
length = 10000
target_names = target_output
y_pred, y_true,x = ml.plot_comparison(x_test_scaled,y_test,y_scaler,
                                                       target_names,model)


mse_vect = list()
for i in range(len(y_pred)):
    mse_vect.append((y_pred[i]-y_true[i])**2)

mse = sum(mse_vect)
plt.figure()
plt.plot(mse_vect)


data_size = 144
training_size = 144
new = np.array(data.solar_irradiance).reshape(8760,2)
new_expand = np.expand_dims(new,axis=0)
new_pred = model.predict(new_expand)
new_pred_rescaled =y_scaler.inverse_transform(new_pred[0])
plt.plot(new_pred_rescaled[0:23]*(data_size/training_size))
plt.plot(data.iloc[:,0][0:23])
