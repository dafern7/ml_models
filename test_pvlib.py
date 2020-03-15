# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 11:59:59 2019

@author: IST_1
"""

import os
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import pvlib
import datetime
import helper
import numpy as np
from pvlib import clearsky, atmosphere, solarposition
from pvlib.location import Location
from pvlib.iotools import read_tmy3
from pvlib.forecast import GFS, NAM, NDFD, HRRR, RAP
import ml

date = pd.date_range(start = '1/1/2018', end = '12/31/2018', freq='H')
solcast = pd.read_csv('new_york_solcast_data.csv')


pv = helper.pv_data(address = 'New York, New York', solar_size_kw_dc = 1000, tilt=0)
dates = pd.to_datetime(solcast.Date)
solcast.Date = dates
#dirint_dni = pvlib.irradiance.dirint(solcast.Ghi,solcast.Zenith,times=pd.to_datetime(date))
solcast['Month'] = solcast.Date.dt.month
solcast['Day'] = solcast.Date.dt.day
solcast['Hour'] = solcast.Date.dt.hour
solcast = solcast.drop(columns=['Date'])
solcast = solcast.reindex(index=np.roll(solcast.index,-4)).reset_index(drop=True)
solcast['solar_output_kw'] = pv.solar_output_kw
solcast = solcast.drop(solcast.loc[(solcast.Dni==0)&(solcast.Dhi==0)&(solcast.Ghi==0)].index)
target_output=['Dhi','Ghi','Dni']
#solcast.loc[solcast.Azimuth<=-1] = solcast.loc[solcast.Azimuth<=-1] + 360


est_dhi = solcast.Ghi-solcast.Dni*np.cos(solcast.Zenith)


model,batch,validation_data,callbacks,x_test_scaled,y_test_scaled,y_test,y_scaler= ml.model_generation(solcast, 
                                                                                                       target_output, 
                                                                                                 batch_size =256, 
                                                                                                 sequence_length=24, 
                                                                                                 training_len=0.75,
                                                                                                 validation_len=0.225)

model=ml.train_model(model,
                      batch,
                      validation_data,
                      x_test_scaled,
                      y_test_scaled,
                      callbacks,
                      epoch_size=25,
                      epoch_steps=100)

ml.save_model(model,'model.h5')

output = ml.load_models('model.h5')

import importlib
importlib.reload(ml)
start_idx = 0
length = 500
target_names = target_output
y_pred, y_true,x = ml.plot_comparison(x_test_scaled,y_test,y_scaler,
                                                       target_names,output,length=length)

