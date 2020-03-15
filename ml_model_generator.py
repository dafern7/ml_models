# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:49:30 2019

@author: Richard_Fu
"""
#model generation function 
import ml
import helper
import pv_data_mod
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pvlib as pv
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding, LSTM
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import keras



solar = pv_data_mod.pv_data(address = 'New York, NY', solar_size_kw_dc = 200, tilt = 20)
solar_1 = pd.read_csv('nyc_data_2017_nsrdb_1.csv')
solar_2 = pd.read_csv('solar_data.csv')
solar_3 = pd.read_csv('new_york_solcast_data.csv')
solar_3 = solar_3.reindex(index=np.roll(solar_3.index,-4)).reset_index(drop=True)
#solar_prev = solar.reindex(index=np.roll(solar.index,1*24)).reset_index(drop=True)
data = solar.drop(columns={'month','day','hour','solar_irradiance'})
#data.solar_irradiance = data.solar_irradiance
data.direct_normal_irradiance = solar.direct_normal_irradiance
data.ambient_temperature = data.ambient_temperature
data['dhi'] = solar_3.Dhi
#data['dni'] = solar_3.Dni
data['ghi'] = solar_3.Ghi

#data['clearsky_dhi'] = solar_1['Clearsky DNI']
#data['clearsky_dni'] = solar_1['Clearsky DHI']
#data['clearsky_ghi'] = solar_1['Clearsky GHI']
data['cloud_type'] = solar_1['Cloud Type']
data['solar_zenith'] = solar_3['Zenith']
data['solar_azimuth'] = solar_3['Azimuth']
data['cloud_opacity'] = solar_3['CloudOpacity']
#data['solar_elevation'] = solar_2['Elevation (refracted)']
#data['albedo'] = solar_1['Surface Albedo']
#solar_zenith = solar_2['Zenith (refracted)']
#solar_azimuth = solar_2['Azimuth angle']

date = pd.date_range(start = '1/1/2018', end = '12/31/2018', freq='H')
new_york = pv.location.Location(latitude=40.71,longitude=-74.01,tz='UTC',altitude=10)
new_york_solar_pos = new_york.get_solarposition(date)
new_york_solar_pos.drop(new_york_solar_pos.tail(1).index,inplace=True)

aoi2 = pv.irradiance.aoi(20,180,new_york_solar_pos.zenith,new_york_solar_pos.azimuth)

#aoi = pv.irradiance.aoi(20,180,solar_zenith,solar_azimuth)

#data['AOI'] = aoi.values
     
#data.solar_output_kw = solar.solar_output_kw
#data = data.drop(data.loc[(data.direct_normal_irradiance==0)].index)
target_output=['solar_output_kw']


model,batch,validation_data,callbacks,x_test_scaled,y_test_scaled,y_test,y_scaler= ml.model_generation(data, 
                                                                                                 target_output, 
                                                                                                 batch_size =256, 
                                                                                                 sequence_length=10, 
                                                                                                 training_len=0.85,
                                                                                                 validation_len=0.1)

model=ml.train_model(model,
                                              batch,
                                              validation_data,
                                              x_test_scaled,
                                              y_test_scaled,
                                              callbacks,
                                              epoch_size=15,
                                              epoch_steps=100)

ml.save_model(model,'nsrdb_model.h5')

output = ml.load_models('nsrdb_model.h5')

import importlib
importlib.reload(ml)
start_idx = 0
length = 500
target_names = target_output
y_pred, y_true,x = ml.plot_comparison(x_test_scaled,y_test,y_scaler,
                                                       target_names,output,length=length)


data_solar_size=1000
new_data_1 = pv_data_mod.pv_data(address = 'Comitan, Mexico', solar_size_kw_dc=data_solar_size, tilt=20)
new_solar_prev = new_data_1.reindex(index=np.roll(solar.index,1*24)).reset_index(drop=True)
new_data = new_data_1.drop(columns={'month','day','hour'})
new_data.solar_irradiance = new_data.solar_irradiance/1000
new_data.direct_normal_irradiance = new_data.direct_normal_irradiance/1000
new_data.ambient_temperature = new_data.ambient_temperature/1000
training_solar_size=20
hour_begin=0
hour_end = 500
plt.figure(figsize=(15,5))
plt.plot(new_data_1.iloc[:,3][hour_begin:hour_end], label='actual')
ml.predict_use_model(output,y_scaler,new_data,target_names,data_solar_size,training_solar_size,hour_begin,hour_end)






