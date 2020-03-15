# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 12:51:48 2019

@author: IST_1
"""

import helper
import pandas as pd
import numpy as np


solar = helper.pv_data(address= 'Lancaster, Massachusetts', solar_size_kw_dc=6032,tilt=23.45)
solar['year'] = 2018
total_output = solar.solar_output_kw.sum()
print(total_output)

date = pd.DataFrame()
date['year'] = solar.year
date['month'] = solar.month
date['day'] = solar.day
date['hour'] = solar.hour

power = 2400
discharge_time = 2
peak_duration = 4
assumed_duration_multiplier = peak_duration/discharge_time 
#this is to calculate utilization based on the number of peak hours and discharge time
#calculated by number of peak hours / discharge_time
eff = 0.9
capacity = power*discharge_time
resource_multiplier = 0.1

solar['date'] = pd.to_datetime(date)
solar = solar.drop(columns={'month','day','hour','year'})
solar['power_out'] = np.nan
solar['utilization'] = np.nan

maxVal = power


spring = solar.loc[(solar.date>=pd.to_datetime('2018-03-01 00:00:00')) & (solar.date <= pd.to_datetime('2018-05-14 23:00:00'))]

summer = solar.loc[(solar.date>=pd.to_datetime('2018-05-15 00:00:00')) & (solar.date <= pd.to_datetime('2018-09-14 23:00:00'))]

fall = solar.loc[(solar.date>=pd.to_datetime('2018-09-15 00:00:00')) & (solar.date <= pd.to_datetime('2018-11-30 23:00:00'))]

winter = solar.loc[((solar.date>=pd.to_datetime('2018-12-01 00:00:00')) & (solar.date <= pd.to_datetime('2018-12-31 23:00:00')))]
winter = winter.append(solar.loc[(solar.date>=pd.to_datetime('2018-01-01 00:00:00')) & (solar.date <= pd.to_datetime('2018-02-28 23:00:00'))])


seasons = [spring,summer,fall,winter]
max_line_limit = 200   

#resilience_multiplier = 1.5 #this will not be considered for this analysis
#spring hours are 17 - 20
spring_multiplier = 3

seasonal_total_injection = list()

seasonal_total_pv = list()
es_credit = list()
pv_credit = list()
utilization = list()
seasonal_total_ratio = list()
for season in seasons:
    injection_ratio = list()
    seasonal_injection = list()
    seasonal_pv = list()
    seasonal_utilization = list()
    season.solar_output_kw = season.solar_output_kw.where(season.solar_output_kw <= maxVal, maxVal)
    for month in set(season.date.dt.month):
        for day in set(season.loc[season.date.dt.month==month].date.dt.day):
            if season.date.dt == spring.date.dt:
                peak_times = solar.loc[(solar.date.dt.month==month)&
                                        (solar.date.dt.day==day)&
                                        (solar.date.dt.hour >= 17)& 
                                        (solar.date.dt.hour <= 20)]
                total_charging_power = (season.loc[(season.date.dt.month==month)&
                                             (season.date.dt.day==day)&
                                             (season.date.dt.hour>=0)&
                                             (season.date.dt.hour <= 16)].solar_output_kw.sum())
            elif season.date.dt == summer.date.dt:
                peak_times = solar.loc[(solar.date.dt.month==month)&
                                        (solar.date.dt.day==day)&
                                        (solar.date.dt.hour >= 15)& 
                                        (solar.date.dt.hour <= 18)]
                total_charging_power = (season.loc[(season.date.dt.month==month)&
                                             (season.date.dt.day==day)&
                                             (season.date.dt.hour>=0)&
                                             (season.date.dt.hour <= 14)].solar_output_kw.sum())
            elif season.date.dt == fall.date.dt:
                peak_times = solar.loc[(solar.date.dt.month==month)&
                                        (solar.date.dt.day==day)&
                                        (solar.date.dt.hour >= 16)& 
                                        (solar.date.dt.hour <= 19)]
                total_charging_power = (season.loc[(season.date.dt.month==month)&
                                             (season.date.dt.day==day)&
                                             (season.date.dt.hour>=0)&
                                             (season.date.dt.hour <= 15)].solar_output_kw.sum())
            elif season.date.dt == winter.date.dt:
                peak_times = solar.loc[(solar.date.dt.month==month)&
                                        (solar.date.dt.day==day)&
                                        (solar.date.dt.hour >= 16)& 
                                        (solar.date.dt.hour <= 19)]
                total_charging_power = (season.loc[(season.date.dt.month==month)&
                                             (season.date.dt.day==day)&
                                             (season.date.dt.hour>=0)&
                                             (season.date.dt.hour <= 15)].solar_output_kw.sum())
            

            
            if total_charging_power >= (capacity/eff):
                battery_out = capacity
            
            else:
                battery_out = total_charging_power*eff
                
            for hour in peak_times.date.dt.hour:
                #subject to max_line_limit
                if peak_times.loc[peak_times.date.dt.hour==hour].solar_output_kw.values <= max_line_limit:
                    if (battery_out/4)+peak_times.loc[peak_times.date.dt.hour==hour].solar_output_kw.values <= max_line_limit:                   
                        peak_times.loc[peak_times.date.dt.hour==hour,'power_out'] = battery_out/4                     
                        
                    else:
                        peak_times.loc[peak_times.date.dt.hour==hour,'power_out'] = max_line_limit-peak_times.loc[peak_times.date.dt.hour==hour].solar_output_kw.values
                else:
                    peak_times.loc[peak_times.date.dt.hour==hour,'power_out'] = 0
                    peak_times.loc[peak_times.date.dt.hour==hour,'solar_output_kw'] = max_line_limit
                
                peak_times.loc[peak_times.date.dt.hour==hour,'utilization'] = peak_times.loc[peak_times.date.dt.hour==hour].power_out/(power/assumed_duration_multiplier)
                
            
            if season.date.dt == spring.date.dt:
                seasonal_multiplier = 1
            elif season.date.dt == summer.date.dt:
                seasonal_multiplier = 3
            elif season.date.dt == fall.date.dt:
                seasonal_multiplier = 1
            elif season.date.dt == winter.date.dt:
                seasonal_multiplier = 3
            else:
                seasonal_multiplier = 0
                
            es_credit.append(peak_times.power_out.sum()/1000 * seasonal_multiplier)
            pv_credit.append(peak_times.solar_output_kw.sum()/1000 * seasonal_multiplier * resource_multiplier)
            seasonal_utilization.append(peak_times.utilization.sum()/peak_duration)
            seasonal_injection.append(peak_times.power_out.sum())
            seasonal_pv.append(peak_times.solar_output_kw.sum()/1000)
            injection_ratio.append((peak_times.power_out.sum())/(peak_times.power_out.sum()+peak_times.solar_output_kw.sum()))
            
            
    seasonal_total_injection.append(sum(seasonal_injection))
    seasonal_total_pv.append(sum(seasonal_pv))
    seasonal_total_ratio.append(sum(injection_ratio)/len(injection_ratio))
    utilization.append(sum(seasonal_utilization)/len(seasonal_utilization))

    
#spring_util =  ((seasonal_total_injection[0])/((capacity/1000)*75))
#summer_util =  ((seasonal_total_injection[1])/((capacity/1000)*123))
#fall_util =  ((seasonal_total_injection[2])/((capacity/1000)*77))
#winter_util =  ((seasonal_total_injection[3])/((capacity/1000)*90))

spring_util =  utilization[0]
summer_util =  utilization[1]
fall_util =  utilization[2]
winter_util =  utilization[3]

weighted_util = (spring_util*1+summer_util*3+fall_util*1+winter_util*3)/8


total_credits = sum(es_credit)+sum(pv_credit)

print(total_credits)
print("Weighted Utilization: " ,weighted_util) 
print("Spring Utilization Rate: ", spring_util)  
print("Summer Utilization Rate: ", summer_util) 
print("Fall Utilization Rate: ", fall_util) 
print("Winter Utilization Rate: ", winter_util) 
print("Percentage of battery output during peak hours: ", sum(seasonal_total_ratio)/len(seasonal_total_ratio))
    
        
        
            
            



        

        
            









