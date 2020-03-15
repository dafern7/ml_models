# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 12:30:40 2019

@author: IST_1
"""

import pvlib as pv
import pandas as pd


solar_2 = pd.read_csv('solar_data.csv')
solar_zenith = solar_2['Zenith (refracted)']
solar_azimuth = solar_2['Azimuth angle']


aoi = pv.irradiance.aoi(20,180,solar_zenith,solar_azimuth)

date = pd.date_range(start = '1/1/2018', end = '1/1/2019', freq='H')
df = pd.DataFrame(date)
df.columns=['date'] 
df.drop(df.tail(1).index,inplace=True)

#new_york = pv.location.Location(latitude=40.71,longitude=-74.01,tz='UTC',altitude=10)
new_york_solar_pos = pv.solarposition.spa_python(date,latitude=42.05,longitude=-72.77,tz='EST',altitude=10)
new_york_solar_pos.drop(new_york_solar_pos.tail(1).index,inplace=True)

aoi2 = pv.irradiance.aoi(20,180,new_york_solar_pos.zenith,new_york_solar_pos.azimuth)