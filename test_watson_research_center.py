# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 10:03:40 2019

@author: Richard_Fu
"""

import helper
import pandas as pd
import ml

df = pd.read_csv('simulation_3164844_hourly_data.csv')
df = df.fillna(0)
df = df.drop(columns={'hour_index','timestamp','horizon_elevation_angle','optimizer_input_power',
                      'optimizer_output_power','dry_bulb_temperature','windspeed','albedo','nameplate_power',
                      'module_mpp_power','module_power','optimal_dc_power','optimal_dc_voltage','actual_dc_voltage',
                      'module_irradiance_derated_power','inverter_overpower_loss','inverter_underpower_loss',
                      'inverter_overvoltage_loss','inverter_undervoltage_loss'})

df = df.div(1000)




target_output = ['actual_dc_power','ac_power','grid_power']

model,batch,validation_data,callbacks,x_test_scaled,y_test_scaled,y_test,y_scaler= ml.model_generation(df, target_output)


model=ml.train_model(model,
                                              batch,
                                              validation_data,
                                              x_test_scaled,
                                              y_test_scaled,
                                              callbacks,
                                              epoch_size=15,
                                              epoch_steps=100)




ml.save_model(model,'watson_ibm_model.h5')

output = ml.load_models('watson_ibm_model.h5')

import importlib
importlib.reload(ml)
start_idx = 0
length = 500
target_names = target_output
y_pred, y_true,x = ml.plot_comparison(x_test_scaled,y_test,y_scaler,
                                                       target_names,output,length=length)


