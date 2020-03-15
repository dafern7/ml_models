# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 15:11:26 2019

@author: Richard_Fu
"""
import pandas as pd
import numpy as np
import ml
import sys, os
from IPython.display import display
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




lat, lon = 42.0548, -72.7704
year = 2018
interval=30
info = pd.DataFrame({'date': pd.date_range(start='1/1/'+str(2018), periods=17520, freq='15T')})

index = pd.date_range(start='1/1/'+str(2018), periods=17520, freq='30T')
info = ml.pv_mod_2(lat,lon,year)

info.index = index

info = info.resample('15T').asfreq()
info = info.drop(columns={'Year','Month','Day','Hour','Minute'})

info = info.interpolate()






df = pd.read_csv('Southwick_ml_dataset.csv')
df = df[0:35039].reset_index(drop=True)
df.index = info.index

solar = df.Southwick_ac_kw
time_index = df.PeriodStart
df = df.drop(columns={'PeriodEnd','PeriodStart','Period','Southwick_ac_kw'})
df = df.reindex(index=np.roll(df.index,-20)).reset_index(drop=True)


df.index = info.index
df.AirTemp = info['Temperature']
df.Dhi = info.DHI
df.Ghi = info.GHI
df.Dni = info.DNI
df['CloudType'] = info['Cloud Type']
df.Zenith = info['Solar Zenith Angle']
df['date'] = time_index
df.date = pd.to_datetime(df.date)
df['month'] = df.date.dt.month
df['day'] = df.date.dt.day
df['hour'] = df.date.dt.hour
df['minute'] = df.date.dt.minute
df['date_of_year'] = df.date.dt.dayofyear

df = df.drop(columns={'date'})

df['maintenance'] = 0
df['solar_output_kw'] = solar
df.loc[df.solar_output_kw == ' - ', ['maintenance']] = 1
df.loc[pd.isnull(df.solar_output_kw), ['maintenance']] = 1
df.loc[df.solar_output_kw == ' - ', ['solar_output_kw']] = 0
df.solar_output_kw = df.solar_output_kw.fillna(0)

df = df.reset_index(drop=True)
df.solar_output_kw = df.solar_output_kw.astype('float')


df.loc[df.Azimuth<=0, ['Azimuth']] = df.Azimuth+360
df.loc[df.solar_output_kw <= 1, ['solar_output_kw']] = 0

tilt = 25
azimuth = 180
solcast_azimuth = df.Azimuth
solcast_azimuth_adjusted = solcast_azimuth-azimuth #adjusted for the azimuth angle of the array

cosines = np.cos(np.deg2rad(df.Zenith))*np.cos(np.deg2rad(tilt))
sines = np.sin(np.deg2rad(df.Zenith))*np.sin(np.deg2rad(tilt))

aoi = np.arccos(cosines+(sines*np.cos(np.deg2rad(solcast_azimuth_adjusted))))
cos_aoi = np.cos(aoi)

#beam of irradiance
E_b = df.Dni * np.cos(aoi)



#ground reflected radiation

albedo = 0.16 #this is the fraction of the ghi that is absorbed by the ground. Taken from a table

E_g = df.Ghi * albedo * ((1-np.cos(np.deg2rad(tilt)))/2)

#diffuse ground irradiance

E_d_iso = df.Dhi * ((1+np.cos(np.deg2rad(tilt)))/2)


#
#solar_new_york = ml.pv_data(address = 'New York, New York', solar_size_kw_dc = 1000, tilt = 20)
#
#Hay/Davies model

E_sc = 1367

doy = df.date_of_year
b = 2*np.pi*(doy/365)
ratio = 1.00011 + 0.034221*np.cos(b) + 0.00128*np.sin(b) + 0.000719*np.cos(2*b) + 0.000077*np.sin(2*b)



E_a = E_sc * ratio

A_i = df.Dni/E_a
R_b = cos_aoi/(np.cos(np.deg2rad(df.Zenith)))

E_d_HayDavies = df.Dhi*(A_i*R_b + (1-A_i)*(1+np.cos(np.deg2rad(tilt)))/2)
E_d_HayDavies.loc[E_d_HayDavies>1e10] = 10
E_d_HayDavies.loc[E_d_HayDavies<-1e10] =0

E_poa = E_b + E_g + E_d_iso


df['E_poa'] = E_poa

plt.figure(figsize=(15,5))
plt.scatter(df.E_poa,df.solar_output_kw)

target_output=['solar_output_kw']

seg_start = 8760*3
seg_end = -1

segment = df[seg_start:seg_end-2]
model,batch,validation_data,callbacks,x_test_scaled,y_test_scaled,y_test,y_scaler= ml.model_generation(segment,
                                                                                                 target_output,
                                                                                                 batch_size=256,
                                                                                                 sequence_length=24,
                                                                                                 training_len=0.9,
                                                                                                 validation_len=0.1)

model=ml.train_model(model,
                      batch,
                      validation_data,
                      x_test_scaled,
                      y_test_scaled,
                      callbacks,
                      epoch_size=20,
                      epoch_steps=100)

ml.save_model(model,'entersolar_model.h5')

output = ml.load_models('entersolar_model.h5')

import importlib
importlib.reload(ml)
start_idx = 0
length = 2000
target_names = target_output
y_pred, y_true,x,mse = ml.plot_comparison(x_test_scaled,y_test,y_scaler,
                                                       target_names,output,
                                                       start_idx = start_idx,
                                                       length=length,
                                                       verbose=True)



solar_pvwatts = helper.pv_data(coordinates=(lat,lon),solar_size_kw_dc = 1750,tilt=25)

index = pd.date_range(start='1/1/'+str(2018), periods=8760, freq='60T')

solar_pvwatts.index = index

solar_pvwatts = solar_pvwatts.resample('15T').asfreq()
#solar_pvwatts = solar_pvwatts.drop(columns={'month','day','hour'})

solar_pvwatts = solar_pvwatts.interpolate()
solar_pvwatts = solar_pvwatts.reset_index(drop=True)
solar_pvwatts = solar_pvwatts[seg_start:seg_end]
segment_pvwatts = solar_pvwatts[-(int(np.ceil(len(segment)-0.9*(len(segment))))+1):-1]
plt.figure(figsize=(15,5))
#plt.plot(segment_pvwatts.index,segment_pvwatts.solar_output_kw,'bo',alpha=0.5)
plt.plot(segment_pvwatts.index,y_pred)
plt.plot(segment_pvwatts.index,(y_true))
plt.legend(['pvwatts','predicted','true'])

plt.xlabel('interval')
plt.ylabel('solar_output_kw')
plt.title('Comparison of Predicted vs PvWatts')
plt.grid(alpha=0.5)

#graphing overpredictions
overpredict = 0
underpredict = 0
accurate = 0

for i in range(0,len(y_pred)):
    if y_pred[i]>y_true[i]:
        overpredict = overpredict + (y_pred[i]-y_true[i])
    elif y_pred[i]<y_true[i]:
        underpredict = underpredict + (y_true[i]-y_pred[i])
    else:
        accurate = accurate+1
