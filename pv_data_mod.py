# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:27:44 2019

@author: Richard_Fu
"""

import pandas as pd
from helper import save_data,load_data
import requests

def pv_data(coordinates=None, address=None, tilt=30,
            solar_size_kw_dc=4500, inverter_size_kw=4500, inverter_efficiency=96,
            system_losses_perc_of_dc_energy=14,
            mount="Fixed (open rack)", module_type="Default [Standard]"):
    """
    :param coordinates: optional, set the coordinates of the location as (latitude, longitude)
    :param address: optional, set the address as a string
    :param tilt: integer, tilt angle default = 40
    :param solar_size_kw_dc: solar size in KW
    :param inverter_size_kw: inverter size in KW
    :param inverter_efficiency: inverter efficiency, default = 96 (96%)
    :param system_losses_perc_of_dc_energy: system losses, default = 14 (14%)
    :param mount: mount type, default = 'Fixed (open rack)'
    :param module_type: module type, default= "Default [Standard]"
    :return: hourly data of expected solar energy
    """
    # TODO:  Figure why inverter_size_kw, export_limit_ac_k_w, export_limit_ac_k_w, dataset,
    #  and annual_solar_degradation_perc_per_year are not implemented
    if coordinates:
        assert isinstance(coordinates, tuple), 'coordinates should be in tuple format'
        latitude = coordinates[0]
        longitude = coordinates[1]
        location = "&" + "lat=" + str(latitude) + "&" + "lon=" + str(longitude)
        tilt = latitude
        verbose_dic = {'Coordinates': coordinates, 'Tilt Angle': int(tilt)}
    elif address:
        location = "&" + "address=" + address
        verbose_dic = {'Address': address, 'Tilt Angle': int(tilt)}
    else:
        raise Exception('either coordinates or address must be input')
    verbose_dic.update({'Solar System Size (kW DC)': solar_size_kw_dc, 'Inverter Size (kW)': inverter_size_kw,
                        'Inverter Efficiency': inverter_efficiency,
                        'System Losses Percentage of DC Energy': system_losses_perc_of_dc_energy})
    save_data(verbose_dic, 'solar_inputs', overwrite=True, description='Solar Inputs to View on Excel')
    # Solar Input Selection Setup and API_Key Setup
    pv_watts_api_key = "VTF48OxZfq7tlP4oriEDVK1qAnpOCPdzl0XGT2c0"
    mount_dict = {"Fixed (open rack)": 0, "Fixed (roof mount)": 1, "1-Axis Tracking": 2,
                  "1-Axis Backtracking": 3, "2-Axis": 4, "Default [Fixed (open rack)]": 0}
    module_type_dict = {"Standard": 0, "Premium": 1, "Thin film": 2, "Default [Standard]": 0}
    array_azimuth_dict = {"135 (SE)": 135, "180 (S)": 180, "225 (SW)": 225}
    dataset = "nsrdb"

    array_azimuth = array_azimuth_dict["180 (S)"]
    mount = mount_dict[mount]
    module_type = module_type_dict[module_type]

    # Over a 25 year life
    annual_solar_degradation_perc_per_year = 0.5 * .01
    # If no export limit then state "No Limit"
    export_limit_ac_k_w = 3000
    variable = 1

    get_link = ''.join(["https://developer.nrel.gov/api/pvwatts/v6.json?" + "api_key=" + pv_watts_api_key,
                        "&" + "system_capacity=" + str(solar_size_kw_dc),
                        "&" + "module_type=" + str(module_type),
                        "&" + "losses=" + str(system_losses_perc_of_dc_energy),
                        "&" + "array_type=" + str(mount),
                        "&" + "tilt=" + str(tilt),
                        "&" + "azimuth=" + str(array_azimuth),
                        location,
                        "&" + "dataset=nsrdb",
                        "&" + "timeframe=hourly",
                        "&" + "inv_eff=" + str(inverter_efficiency), ]
                       )
    result = requests.get(get_link)
    data = result.json()
    outs_w = data["outputs"]["ac"]
    outs_kw = [out / 1000 for out in outs_w]
    outs_poa = data["outputs"]["poa"]
    outs_dn = data["outputs"]["dn"]
    outs_tamb = data["outputs"]["tamb"]
    outs_poa = pd.Series(outs_poa)
    outs_kw = pd.Series(outs_kw)
    outs_dn = pd.Series(outs_dn)
    outs_tamb = pd.Series(outs_tamb)
    df = load_data('hours_in_year')
    df['solar_output_kw'] = outs_kw.values
    df['solar_irradiance'] = outs_poa.values
    df['direct_normal_irradiance'] = outs_dn.values
    df['ambient_temperature'] = outs_tamb.values *1.8000+32

    return df

df = pv_data(address = 'New York, NY',solar_size_kw_dc=144)