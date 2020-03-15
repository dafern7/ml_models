# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:37:04 2019

@author: IST_1
"""

import csv
import pandas as pd
import numpy as np
import helper

txt_file = r"s3159989.txt"
csv_file = r"solar_data.csv"



with open(txt_file, 'r') as infile, open(csv_file, 'w') as outfile:
     stripped = (line.strip() for line in infile)
     lines = (line.split(",") for line in stripped if line)
     writer = csv.writer(outfile)
     writer.writerows(lines)
     
     
df = pd.read_csv('solar_data.csv')
tilt=-20
angle_of_array=180

aoi = np.arccos((np.cos(df['Zenith (refracted)'])*np.cos(tilt))+(np.sin(df['Zenith (refracted)'])*np.sin(tilt)*np.cos(df['Azimuth angle']-angle_of_array)))

dni_df = helper.pv_data(address='New York, NY', solar_size_kw_dc=1230,tilt=20)

dni_df.direct_normal_irradiance = dni_df.direct_normal_irradiance*aoi