	# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 11:40:32 2019

@author: Richard_Fu
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from darksky import forecast
from datetime import datetime as dt
from calendar import monthrange
import os
from sklearn.preprocessing import MinMaxScaler, Normalizer, MaxAbsScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from keras.models import model_from_json
from tensorflow import keras

import keras.backend as K


def punish_overprediction(true,pred):
    e = pred - true

    overprediction = K.greater(e,0)
    underprediction = K.less(e,0)
    overprediction = K.cast(overprediction, K.floatx())
    underprediction = K.cast(underprediction, K.floatx())
    prediction = overprediction*6 + underprediction

    #mean absolute error here
    return K.mean(prediction*K.abs(e))

keras.losses.custom_loss = punish_overprediction

def batch_generator(batch_size, sequence_length, x_data,y_data,training_number,x_train_scaled,y_train_scaled):

    while True:
    #initialize batches, target and training set size
        x_batch = np.zeros(shape=(batch_size, sequence_length, x_data.shape[1]), dtype=np.float16)
        y_batch = np.zeros(shape=(batch_size, sequence_length, y_data.shape[1]), dtype=np.float16)

        for i in range(batch_size):

            #want to start at random times
            idx = np.random.randint(int(training_number) - sequence_length)

            #want a days worth of data (sequence length 24)
            x_batch[i] = x_train_scaled[idx:idx+sequence_length]
            y_batch[i] = y_train_scaled[idx:idx+sequence_length]

        #need to create a generator for the keras model as well as batch generator
        yield (x_batch, y_batch)


#takes in data, plots out both for test data
def plot_comparison(x_test_scaled,y_test,y_scaler,target_names,model,start_idx=0,length=3000,verbose=False):

    x = x_test_scaled
    y_actual = y_test

    end_idx = start_idx + length

    x = x[start_idx:end_idx]
    y_actual = y_actual[start_idx:end_idx]

    # need to generate 3 dimensional tensors for the model to predict, since our test data
    #was originally 3 dimensional tensors
    x = np.expand_dims(x, axis=0)

    # Use the model to predict the output-signals.
    y_pred = model.predict(x)


    #rescale back to original, downconvert to a 2d array so it can be graphed
    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])

    # For each output-signal.
    for dimension in range(len(target_names)):

        #plotting different graphs based on each target dimension
        predicted = y_pred_rescaled[:, dimension]
        true_val = y_actual[:, dimension]
        plt.figure(figsize=(15,5))
        # plot and compare the two signals.
        plt.plot(true_val, label='actual')
        plt.plot(predicted, label='predicted')
        plt.ylabel(target_names[dimension])
        plt.xlabel('Duration')
        plt.legend()
        plt.show()

    mse = 0
    if verbose:
        mse_vect = list()
        for i in range(len(y_pred_rescaled)):
            mse_vect.append((abs(y_pred_rescaled[i]-y_actual[i])))

        mse = sum(mse_vect)/len(mse_vect)
        plt.figure()
        plt.plot(mse_vect)
        plt.show()

    return y_pred_rescaled, y_actual, x, mse


def call_darksky(latcoord,longcoord,year):
    temp=[]
    dewpoint=[]
    humid=[]
    press=[]
    cloud=[]
    uv=[]
    precip_prob=[]
    key = '75bd65ebbfae8f7a4f79c45950569da8'
    for i in range(1,13):
        for j in range(1,monthrange(year,i)[1]+1):
            t = dt(year, i, j, 12).isoformat()
            past_year = forecast(key, latcoord, longcoord, time=t)
            for hour in past_year.hourly:
                try:
                    temp.append(hour.temperature)
                    dewpoint.append(hour.dewPoint)
                    humid.append(hour.humidity)
                    press.append(hour.pressure)
                    cloud.append(hour.cloudCover)
                    uv.append(hour.uvIndex)
                    precip_prob.append(hour.precipProbability)
                except Exception:
                    temp.append([0])
                    dewpoint.append([0])
                    humid.append([0])
                    press.append([0])
                    cloud.append([0])
                    uv.append([0])
                    precip_prob.append([0])

    df = pd.DataFrame(pd.date_range(start='1/1/2018', end='1/1/2019',freq='H'))
    df = df.drop(df.index[-1],axis=0)
    df['temp'] = temp
    df['dewpoint'] = dewpoint
    df['humidity']  = humid
    df['pressure'] = press
    df['cloud_cover'] = cloud
    df['uv_index'] = uv
    df['precipitation_probability'] = precip_prob
    return df


def pv_mod_2(lat,lon,year):
    api_key = 'QvZZnJlwMlRmxmdxZRcNBxCCxoRLdXaB0aSwN9T6'
    attributes = 'ghi,dhi,dni,cloud_type,air_temperature,solar_zenith_angle'
    year = str(year)
    leap_year = 'false'
    interval = '30'
    utc = 'false'
    name = 'Richard+Fu'
    reason_for_use = 'beta+testing'
    your_affiliation = 'Integrated+Storage+Technologies'
    email = 'richard@integratedstoragetech.com'
    mailing_list = 'false'
    url = 'http://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=name, email=email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=api_key, attr=attributes)
    info = pd.read_csv(url, skiprows=2)
    return info




def model_generation(data, target_output, batch_size =256, sequence_length=24, training_len=0.7,validation_len=0.1):
    #target variables to select (this will determine the number of output neurons)
    target_df = data[target_output]

    #take the rest of the dataframe
    features = data.drop(columns=target_output).values

    x = features
    y = target_df.values
    training_num = int(training_len*len(x))
    val_num = int(validation_len*len(x))

    x_train = x[0:training_num]
    x_val = x[training_num:training_num+val_num]
    x_test = x[training_num:]

    y_train = y[0:training_num]
    y_val = y[training_num:training_num+val_num]
    y_test = y[training_num:]

    x_scaler = MaxAbsScaler()
    x_train_scaled = x_scaler.fit_transform(x_train)
    x_val_scaled = x_scaler.fit_transform(x_val)
    x_test_scaled = x_scaler.fit_transform(x_test)

    y_scaler = MaxAbsScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_val_scaled = y_scaler.fit_transform(y_val)
    y_test_scaled = y_scaler.fit_transform(y_test)



    batch = batch_generator(batch_size = batch_size,sequence_length=sequence_length,x_data=x,y_data=y,
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
    model.add(LSTM(units=256,return_sequences=True,input_shape=(None,x.shape[1],)))

    model.add(Dense(256,activation='tanh',kernel_regularizer=keras.regularizers.l2(l=0.1)))
#    model.add(Dense(128,activation = 'tanh',kernel_regularizer=keras.regularizers.l2(l=0.1)))
#    model.add(LSTM(units=64,return_sequences=True,input_shape=(None,64,)))
#    model.add(Dense(512,activation = 'tanh',kernel_regularizer=keras.regularizers.l2(l=0.1)))
#    model.add(Dense(256,activation = 'tanh',kernel_regularizer=keras.regularizers.l2(l=0.1)))
#    model.add(LSTM(units=128,return_sequences=True,input_shape=(None,256,)))
#    model.add(Dense(256,activation = 'tanh'))
#    model.add(Dense(128,activation = 'tanh'))
#    model.add(Dense(64,activation = 'tanh'))
#    model.add(Dense(32,activation = 'tanh'))

    #mapping down to target_output output units from the 512 hidden layer units using tanh function
    model.add(Dense(y.shape[1],activation='tanh',kernel_regularizer=keras.regularizers.l2(l=0.01)))
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    rms = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.01)
    #loss function mean absolute error, adam optimizer



    model.compile(optimizer=adam , loss=punish_overprediction,  metrics=['accuracy'])


    #callbacks are for logging process
    path_checkpoint = '23_checkpoint.keras'
    checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                          monitor='val_loss',
                                          verbose=1,
                                          save_weights_only=True,
                                          save_best_only=True)

    early_stopping = EarlyStopping(monitor='val_loss',
                                            patience=4, verbose=1)


    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.1,
                                           min_lr=1e-4,
                                           patience=0,
                                           verbose=1)

    callbacks = [early_stopping,
                 checkpoint,
                 reduce_lr]

    return model,batch,validation_data, callbacks, x_test_scaled, y_test_scaled,y_test,y_scaler


def train_model(model,batch,validation_data,x_test_scaled,y_test_scaled,callbacks,epoch_size=2,epoch_steps=100):

    model.fit_generator(generator=batch,
                    epochs=epoch_size,
                    steps_per_epoch=epoch_steps,
                    validation_data=validation_data,
                    callbacks=callbacks)
    return model

def save_model(model, model_filename):
    '''
    saves model into model.h5 for use in load model
    '''
    model_json = model.to_json()
    with open(model_filename, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")

    print("saved model")

def load_models(model_filename):
    '''
    the filename should be in .h5 file format
    '''
    json_file = open(model_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    output =  tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    output.load_weights("model.h5")
    print("loaded data!")
    return output


def predict_use_model(model,y_scaler,new_data,target_names,data_solar_size,training_solar_size,hour_begin,hour_end):
    '''
    uses model to predict upcoming data
    '''
    new = np.array(new_data.drop(columns=target_names))
    new = new.reshape(new.shape[0],new.shape[1])
    new_expand = np.expand_dims(new,axis=0)
    new_pred = model.predict(new_expand)
    new_pred_rescaled =y_scaler.inverse_transform(new_pred[0])

#    plt.plot(new_pred_rescaled[hour_begin:hour_end]*(data_solar_size/training_solar_size),label='predicted')
#    plt.xlabel('Duration')
#    plt.ylabel('solar_output_kw')
#    plt.legend()

    return new_pred_rescaled
