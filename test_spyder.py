# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 17:01:04 2019

@author: Richard_Fu
"""

import datetime
import helper
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import importlib
import op_models
importlib.reload(helper)
importlib.reload(op_models)
from sklearn.ensemble import RandomForestClassifier
import numpy as np


df = pd.read_excel('File 2.xlsx')
df = df[['Device name','Unnamed: 8']]
df = df.drop([0,1,2,3,4,5,6])
df = df.rename(columns={'Device name': 'date', 'Unnamed: 8': 'original_building_power_kw'})
df.date = pd.to_datetime(df.date)
df.original_building_power_kw = df.original_building_power_kw/1000
df['original_building_energy_kwh'] = df.original_building_power_kw/4
df = df.sort_values(by='date')
df = df.reset_index(drop=True)
solar = helper.pv_data(address='Chirin Gita, Mexico',solar_size_kw_dc=80)
df = helper.add_solar_to_df(df,solar,4)
#f['y'] = 'yes'
#df['target'] = df['y'].apply(lambda x: 1 if x=='yes' else 0)

df.isnull().mean().sort_values(ascending=False)*100
#df.drop('y',axis=1,inplace=True)
df.original_building_power_kw = df.original_building_power_kw.astype(float)
df.original_building_energy_kwh = df.original_building_energy_kwh.astype(float)
df.power_post_solar_kw = df.power_post_solar_kw.astype(float)
df.energy_post_solar_kwh = df.energy_post_solar_kwh.astype(float)
#df['month'] = df.date.dt.month.astype(float)
#df['day'] = df.date.dt.day.astype(float)
#df['hour'] = df.date.dt.hour.astype(float)


corr = df.corr()
#sns.heatmap(corr,
#            xticklabels=corr.columns,
#            yticklabels=corr.columns)


#df['date'] = list(range(len(df)))
features = df[df.columns.difference(['date'])]
labels = df['date']
vif = pd.DataFrame()
vif["Features"] = features.columns

from sklearn.model_selection import train_test_split

#df.reshape()
train, test = train_test_split(df, test_size = 0.4)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

features_train = train[list(vif['Features'])]
label_train = train['date']
features_test = test[list(vif['Features'])]
label_test = test['date']


clf = RandomForestClassifier()

clf.fit(features_train,label_train)

pred_train = clf.predict(features_train)
pred_test = clf.predict(features_test)

from sklearn.metrics import accuracy_score
accuracy_train = accuracy_score(pred_train,label_train)
accuracy_test = accuracy_score(pred_test,label_test)




#from sklearn import metrics
#fpr, tpr, _ = metrics.roc_curve(np.array(label_train), clf.predict_proba(features_train)[:,1])
#auc_train = metrics.auc(fpr,tpr)

#fpr, tpr, _ = metrics.roc_curve(np.array(label_test), clf.predict_proba(features_test)[:,1])
#auc_test = metrics.auc(fpr,tpr)

valid = pd.crosstab(label_train,pd.Series(pred_train),rownames=['ACTUAL'],colnames=['PRED'])