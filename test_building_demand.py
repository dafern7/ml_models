# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 12:41:23 2019

@author: IST_1
"""


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

df = pd.read_excel('test_building_demand.xlsx')
df = df[0:95000]
df = df.drop(columns={'Unnamed: 0','Channel Name','Meter Serial Number'})

df = df[df['Read Value'] != 0]

df['date'] = pd.to_datetime(df['Read End Time'])

df['month'] = df.date.dt.month
df['day'] = df.date.dt.day
df['hour'] = df.date.dt.hour
df['minute'] = df.date.dt.minute

df = df.drop(columns={'date','Read End Time'})

target_output=['Read Value']


model,batch,validation_data,callbacks,x_test_scaled,y_test_scaled,y_test,y_scaler= ml.model_generation(df, 
                                                                                                 target_output, 
                                                                                                 batch_size=256, 
                                                                                                 sequence_length=20, 
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

ml.save_model(model,'building_demand.h5')
 
output = ml.load_models('building_demand.h5')

import importlib
importlib.reload(ml)
start_idx = 0
length = 500
target_names = target_output
y_pred, y_true,x,mse = ml.plot_comparison(x_test_scaled,y_test,y_scaler,
                                                       target_names,output,
                                                       start_idx = start_idx,
                                                       length=length,
                                                       verbose=True)