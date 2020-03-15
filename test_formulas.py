# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 14:14:55 2019

@author: Richard_Fu
"""

#testing out some pv estimations based on Sandia Laboratories PvLib 

import pandas as pd
import numpy as np
import ml
from matplotlib import pyplot as plt

solar_new_york = ml.pv_data(address = 'New York, New York', solar_size_kw_dc = 1000, tilt = 20)
solcast_data = pd.read_csv('new_york_solcast_data.csv')
solcast_data = solcast_data.drop(columns={'Period'})
solcast_data = solcast_data.rename(columns={'PeriodStart':'date'})
solcast_data.date = pd.to_datetime(solcast_data.date)
solcast_data['Month'] = solcast_data.date.dt.month
solcast_data['Day'] = solcast_data.date.dt.day
solcast_data['Hour'] = solcast_data.date.dt.hour
solcast_data['date_of_year'] = solcast_data['date'].dt.dayofyear

solcast_data = solcast_data.reindex(index=np.roll(solcast_data.index,-4)).reset_index(drop=True)
#angle of incidence calculations

tilt = 20
azimuth = 180
solcast_azimuth = solcast_data.Azimuth
solcast_azimuth.loc[solcast_azimuth<=0] = 360 + solcast_azimuth
solcast_azimuth_adjusted = solcast_azimuth-azimuth #adjusted for the azimuth angle of the array

cosines = np.cos(np.deg2rad(solcast_data.Zenith))*np.cos(np.deg2rad(tilt))
sines = np.sin(np.deg2rad(solcast_data.Zenith))*np.sin(np.deg2rad(tilt))

aoi = np.arccos(cosines+(sines*np.cos(np.deg2rad(solcast_azimuth_adjusted))))
cos_aoi = np.cos(aoi)

#beam of irradiance
E_b = solar_new_york.direct_normal_irradiance * np.cos(aoi)


#ground reflected radiation

albedo = 0.18 #this is the fraction of the ghi that is absorbed by the ground. Taken from a table

E_g = solcast_data.Ghi * albedo * ((1-np.cos(np.deg2rad(tilt)))/2)

#diffuse ground irradiance

E_d_iso = solcast_data.Dhi * ((1+np.cos(np.deg2rad(tilt)))/2)



solar_new_york = ml.pv_data(address = 'New York, New York', solar_size_kw_dc = 1000, tilt = 20)

#Hay/Davies model

E_sc = 1367

doy = solcast_data.date_of_year
b = 2*np.pi*(doy/365)
ratio = 1.00011 + 0.034221*np.cos(b) + 0.00128*np.sin(b) + 0.000719*np.cos(2*b) + 0.000077*np.sin(2*b)



E_a = E_sc * ratio

A_i = solar_new_york.direct_normal_irradiance/E_a
R_b = cos_aoi/(np.cos(np.deg2rad(solcast_data.Zenith)))

E_d_HayDavies = solcast_data.Dhi*(A_i*R_b + (1-A_i)*(1+np.cos(np.deg2rad(tilt)))/2)
E_d_HayDavies.loc[E_d_HayDavies>1e10] = 10
E_d_HayDavies.loc[E_d_HayDavies<-1e10] = 10

E_poa = E_b + E_g + E_d_HayDavies



output = ml.load_models('model.h5')

data_solar_size=1000
target_names= ['solar_output_kw']
target_output = target_names


new_data_1 = ml.pv_data(address = 'New York, New York', solar_size_kw_dc=data_solar_size, tilt=20)
#new_solar_prev = new_data_1.reindex(index=np.roll(solar.index,1*24)).reset_index(drop=True)
new_data = new_data_1.drop(columns={'month','day','hour'})
new_data.solar_irradiance = E_poa/1000
new_data.direct_normal_irradiance = solar_new_york.direct_normal_irradiance/1000
new_data.ambient_temperature = new_data.ambient_temperature/1000
training_solar_size=1000
hour_begin=0
hour_end = 500
plt.figure(figsize=(15,5))
plt.plot(solar_new_york.iloc[:,3][hour_begin:hour_end], label='actual')




model,batch,validation_data, callbacks, x_test_scaled, y_test_scaled,y_test,y_scaler = ml.model_generation(new_data, 
                                                                                                           target_output, 
                                                                                                           batch_size =256, 
                                                                                                           sequence_length=24, 
                                                                                                           training_len=0.7,
                                                                                                           validation_len=0.1)

predicted = ml.predict_use_model(output,y_scaler,new_data,target_names,data_solar_size,training_solar_size,hour_begin,hour_end)





