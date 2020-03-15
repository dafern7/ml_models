import numpy as np
from pyomo.environ import *
import pandas as pd
import datetime
from op_helpers import model_to_df, filter_demand_rates, microgrid_model_to_df


def base_load(df, power, capacity, eff=.8, compensation_rate=0, reserve_rate=0, base_load_window=(0,23)):
    """
    Optimize the charge/discharge behavior of a battery storage unit over a
    full year. Assume perfect foresight of electricity prices. The battery
    has a discharge constraint equal to its storage capacity and round-trip
    efficiency of 80%.

    Parameters
    ----------


    :param df : solar dataframe with hourly data
    :param power: power of the battery in kw
    :param capacity: capacity of the battery in kwh
    :param eff: optional, round trip efficiency
    :param compensation_rate: $/kWh for energy injected to the grid
    :param reserve_rate: $/kWh for energy stored
    :param base_load_window: range of hours of the base load, default (0,23)

    :return dataframe
        hourly state of charge, charge/discharge rates
    """
    # assertions
    assert 'solar_output_kw' in df.columns, 'no solar_output_kw column'
    assert 'hour' in df.columns, 'no hour column in the dataframe'
    assert isinstance(base_load_window, tuple), 'base_load_window must be tuple'

    # variable for converting power to energy (since this is an hour by hour dispatch, dih =1)
    dih = 1
    # index must start with 0
    if df.index[0] != 0:
        df.reset_index(drop=True, inplace=True)

    # Define model sets and parameters
    model = ConcreteModel()
    model.T = Set(doc='index', initialize=(df.index + 1).tolist(), ordered=True)
    model.Rmax = Param(initialize=power, doc='Max rate of power flow (kW) in or out')
    model.Smax = Param(initialize=capacity, doc='Max storage (kWh)')
    # create solar vector
    solar_dic = dict(zip(model.T.keys(), df.solar_output_kw.tolist()))
    model.Solar = Param(model.T, initialize=solar_dic, doc='solar output')

    # variables: Charge, discharge, binary variable, state of charge, Pmax
    model.Ein = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.Eout = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.X = Var(model.T, domain=Binary, initialize=0.0)
    model.S = Var(model.T, bounds=(0, model.Smax), initialize=model.Smax)

    # charge the battery from solar power only
    def only_solar_constraint(model, t):
        return model.Ein[t] <= model.Solar[t]
    model.only_solar_constraint = Constraint(model.T, rule=only_solar_constraint)

    # Pmax Constraint, not active
    # TODO: Ask if there are any power constraints for the project
    def power_constraint(model, t):
        rate_cat = df.demand_rate_category.iloc[t - 1]
        return (model.Output[t] + model.Ein[t] - model.Eout[t]) <= model.Pmax[rate_cat]

    #model.power_constraint = Constraint(model.T, rule=power_constraint)

    # Storage State
    def storage_state(model, t):
        'Storage changes with flows in/out and efficiency losses'
        # Set first hour state of charge to half of max
        if t == model.T.first():
            return model.S[t] == 0
        else:
            return (model.S[t] == (model.S[t - 1]
                                   + dih * (model.Ein[t-1] * np.sqrt(eff))
                                   - dih * (model.Eout[t-1] / np.sqrt(eff))))

    model.charge_state = Constraint(model.T, rule=storage_state)

    def base_load_constraint(model, t):
        index = t-1
        window = list(range(base_load_window[0], base_load_window[1]))
        if df.iloc[index].hour in window:
            return model.Solar[t] + model.Eout[t] - model.Ein[t] == model.Solar[t+1] + model.Eout[t+1] - model.Ein[t+1]
        elif df.iloc[index].hour == window[-1]+1:
            return Constraint.Skip
        else:
            return model.Solar[t] + model.Eout[t] - model.Ein[t] == 0

    model.base_load_constraint = Constraint(model.T, rule=base_load_constraint)

    def discharge_constraint(model, t):
        """Maximum discharge within a single hour"""
        return model.Eout[t] <= model.Rmax * model.X[t]

    model.discharge = Constraint(model.T, rule=discharge_constraint)

    def charge_constraint(model, t):
        """Maximum charge within a single hour"""
        return model.Ein[t] <= model.Rmax * (1 - model.X[t])

    model.charge = Constraint(model.T, rule=charge_constraint)

    # Without a constraint the model would discharge in the final hour
    # even when SOC was 0.
    def positive_charge(model, t):
        'Limit discharge to the amount of charge in battery, including losses'
        return model.Eout[t] <= model.S[t]*np.sqrt(eff) / dih
    model.positive_charge = Constraint(model.T, rule=positive_charge)

    # Define the battery income
    # Income:
    energy_income = sum(compensation_rate * (model.Eout[t] + model.Solar[t] - model.Ein[t]) for t in model.T)
    reserve_income = sum(reserve_rate * model.S[t] for t in model.T)

    income = energy_income + reserve_income
    model.P = income
    model.objective = Objective(expr=income, sense=maximize)

    # Solve the model
    solver = SolverFactory('gurobi')
    solver.solve(model)

    return model_to_df(model), model


def peak_shave(df, power, capacity, eff=.8, itc=False, project_type='solar+storage', export=False):
    """
    Optimize expenses resultant from peak shaving

    Parameters
    ----------


    :param df : input dataframe with original building power and solar
    :param power: power of the battery in kw
    :param capacity: capacity of the battery in kwh
    :param eff: optional, round trip efficiency
    :param itc: optional, whether the itc incentive applies
    :param project_type: solar+storage or storage only
    :param export: optional, whether system is allowed to send power to the grid

    :return dataframe
        hourly state of charge, charge/discharge rates
    """
    assert 'date' in df.columns, 'dataframe has no date column'
    assert isinstance(df.date.iloc[0], datetime.date), 'date is not a datetime data type'

    # index must start with 0
    df.reset_index(drop=True, inplace=True)

    # dih: data interval hours
    dih = (df.date.iloc[1] - df.date.iloc[0])
    dih = dih.total_seconds() / 3600
    dih = abs(dih)
    print(dih)
    num_col = 0
    
    for colum in df.columns: 
        if 'demand_rate_category' in colum:
            num_col = num_col+1
            
    demand_categories, demand_rates_dic = filter_demand_rates(df,num_columns=num_col)

    model = ConcreteModel()
    model.T = Set(doc='index', initialize=(df.index + 1).tolist(), ordered=True)
    model.col = Set(initialize=list(range(1,num_col+1)),ordered=True)
    model.Rmax = Param(initialize=power, doc='Max rate of power flow (kW) in or out')
    model.DemandCategories = Set(initialize=demand_categories, doc='demand rate categories', ordered=True)
    model.Smax = Param(initialize=capacity, doc='Max storage (kWh)')
    # create solar vector
    if project_type == 'solar+storage':
        solar_dic = dict(zip(model.T.keys(), df.solar_output_kw.tolist()))
        model.Solar = Param(model.T, initialize=solar_dic, doc='solar output')
        df['output'] = df.power_post_solar_kw
    else:
        df['output'] = df.original_building_power_kw
    # output vector (without storage)
    model.Output = Param(model.T, initialize=dict(zip(model.T.keys(), df.output.tolist())),
                         doc='original building power or post solar')

    # Rates

    model.Erate = Param(model.T, initialize=dict(zip(list(model.T), df.energy_rate.tolist())), doc='ENERGY RATES')
    model.DemandRates = Param(model.DemandCategories, initialize=demand_rates_dic)

    # variables: Charge, discharge, binary variable, state of charge, Pmax
    model.Ein = Var(model.T, domain=NonNegativeReals, initialize=100.0)
    model.Eout = Var(model.T, domain=NonNegativeReals, initialize=100.0)
    model.X = Var(model.T, domain=Binary, initialize=0.0)
    model.S = Var(model.T, bounds=(0, model.Smax), initialize=0)
    model.Pmax = Var(model.DemandCategories, domain=NonNegativeReals, initialize=600)

    if itc:
        assert 'solar_output_kw' in df.columns, "can't add ITC without solar"
        model.SolarOut = Param(model.T, initialize=dict(zip(model.T.keys(), df.solar_output_kw.tolist())), doc='solar')

        def itc_constraint(model, t):
            return model.Ein[t] <= model.SolarOut[t]
        model.itc_constraint = Constraint(model.T, rule=itc_constraint)

    if not export:
        def no_export(model, t):
            return model.Output[t] + model.Ein[t] - model.Eout[t] >= 0

        model.no_export = Constraint(model.T, rule=no_export)

    # Pmax Constraint
    if len(model.col)==1:
        def power_constraint(model, t):
            rate_cat = df.demand_rate_category.iloc[t - 1]
            return (model.Output[t] + model.Ein[t] - model.Eout[t]) <= model.Pmax[rate_cat]

        model.power_constraint = Constraint(model.T, rule=power_constraint)
        
    else:
        def column_constraint(model,t,col):
            cat_col = df['demand_rate_category'+str(col)]
            rate_cat = cat_col.iloc[t-1]
            #print(model.Output[t]+model.Ein[t].value-model.Eout[t].value)
            #print(model.Pmax[rate_cat].value)
            return(model.Output[t]+model.Ein[t]-model.Eout[t] <= model.Pmax[rate_cat])

        model.column_constraint = Constraint(model.T,model.col,rule=column_constraint)
    

    # Storage State
    def storage_state(model, t):
        'Storage changes with flows in/out and efficiency losses'
        # Set first hour state of charge to half of max
        if t == model.T.first():
            return model.S[t] == 0
        else:
            return (model.S[t] == (model.S[t - 1]
                                   + dih * (model.Ein[t - 1] * np.sqrt(eff))
                                   - dih * (model.Eout[t - 1] / np.sqrt(eff))))

    model.charge_state = Constraint(model.T, rule=storage_state)

    def discharge_constraint(model, t):
        """Maximum discharge within a single hour"""
        return model.Eout[t] <= model.Rmax * model.X[t]

    model.discharge = Constraint(model.T, rule=discharge_constraint)

    def charge_constraint(model, t):
        """Maximum charge within a single hour"""
        return model.Ein[t] <= model.Rmax * (1 - model.X[t])

    model.charge = Constraint(model.T, rule=charge_constraint)

    # Without a constraint the model would discharge in the final hour
    # even when SOC was 0.
    def positive_discharge(model, t):
        'Limit discharge to the amount of charge in battery, including losses'
        return model.Eout[t] <= model.S[t]*np.sqrt(eff) / dih
    model.positive_discharge = Constraint(model.T, rule=positive_discharge)
    
    def positive_charge(model, t):
        'Limit discharge to the amount of charge in battery, including losses'
        return model.Ein[t] <= (model.Smax - model.S[t]) *np.sqrt(eff) / dih
    model.positive_charge = Constraint(model.T, rule=positive_charge)

    energy_expenses = dih * sum(
        model.Erate[t] * (model.Output[t] + model.Ein[t] - model.Eout[t]) for t in model.T)
    demand_expenses = sum(model.Pmax[key] * model.DemandRates[key] for key in model.Pmax.keys())
    expenses = energy_expenses + demand_expenses
    model.objective = Objective(expr=expenses, sense=minimize)

    # Solve the model
    solver = SolverFactory('gurobi')
    solver.solve(model)

    return model_to_df(model,project_type), model

def microgrid(df,large_system_size,eff=0.9,dod=0.85,power_val=2000,capacity_val=700,generator_size=0,generator_cost=0.5):
    """
    For optimization of the capacity and power requirements of a battery set in a microgrid environment with no connection to grid.
    Minimizes the expenses resulting from capacity and power requirements.

    Parameters
    ----------


    :param df : input dataframe
    :param large_system_size: size of the system used to initialize the model
    :param eff: optional, round trip efficiency
    :param dod: optional, depth of discharge
    :param power_val: power cost of the battery in $/kw
    :param capacity_val: capacity cost of the battery in $/kwh
    :param generator_size: optional, using an additional generator may lower the necessary battery size
    :param generator_cost: cost of the generator in $/kw

    :return dataframe:
        hourly state of charge, charge/discharge rates
    :return capacity:
        estimated capacity of the battery
    :return power:
        estimated power of the battery
    :return solar_cost:
        estimated cost of the solar size
    :return alpha:
        fraction of the size of the initial system used for the model -> represents the proper size of the system
    :return system_size:
        system size
    """
    
    
    
    assert 'date' in df.columns, 'dataframe has no date column'
    assert isinstance(df.date.iloc[0], datetime.date), 'date is not a datetime data type'

    # index must start with 0
    df.reset_index(drop=True, inplace=True)

    # dih: data interval hours
    dih = (df.date.iloc[1] - df.date.iloc[0])
    dih = dih.total_seconds() / 3600
    dih = abs(dih)
    #print(dih)
    large_system_cost = large_system_size * power_val

    
    model = ConcreteModel()
    model.T = Set(doc='index', initialize=(df.index + 1).tolist(), ordered=True)
    
    solar_dic = dict(zip(model.T.keys(), df.solar_output_kw.tolist()))
    model.Solar = Param(model.T, initialize=solar_dic, doc='solar output')
    building_power_dic = dict(zip(model.T.keys(), df.original_building_power_kw.tolist()))
    model.Building_Power = Param(model.T, initialize=building_power_dic, doc='original building power')
    model.generator = Param(initialize=generator_size, doc='adding an additional generator')
    #df['output'] = df.power_post_solar_kw
    
    #model.Output = Param(model.T, initialize=dict(zip(model.T.keys(), df.original_building_power_kw.tolist())),
                         #doc='original building power')
    #model.Smax=Param(model.T,initialize=20000,doc='max capacity')
    
    
    model.Ein = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.Eout = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.X = Var(model.T, domain=Binary, initialize=0.0)
    model.Smax = Var(domain=NonNegativeReals, initialize=0.0)
    model.Rmax = Var(domain=NonNegativeReals, initialize=0.0)
    model.alpha = Var(bounds=(0,1), initialize = 0.5)        
    model.S = Var(model.T, domain=NonNegativeReals, initialize=0.0)
    model.diss = Var(model.T, bounds=(-model.generator,0), initialize=0.0)
    
    def power_constraint(model, t):
        return ((model.Building_Power[t] - model.Solar[t]*model.alpha + model.Ein[t] - model.Eout[t] + model.diss[t]) == 0)
    model.power_constraint = Constraint(model.T, rule=power_constraint)
    
    def solar_constraint(model, t):
        return (model.Ein[t] <= model.Solar[t]*model.alpha)
    model.solar_constraint = Constraint(model.T, rule=solar_constraint)
               
    def storage_state(model, t):
        'Storage changes with flows in/out and efficiency losses'
        # Set first hour state of charge to half of max
        if t == model.T.first():
            return model.S[t] == model.Smax*dod
        else:
            return (model.S[t] == (model.S[t - 1]
                                   + dih * (model.Ein[t - 1] * np.sqrt(eff))
                                   - dih * (model.Eout[t - 1] / np.sqrt(eff))))
    model.charge_state = Constraint(model.T, rule=storage_state) 
        
    
    def discharge_constraint(model, t):
        """Maximum discharge within a single hour"""
        return model.Eout[t] <= model.Rmax
    model.discharge = Constraint(model.T, rule=discharge_constraint)

    def charge_constraint(model, t):
        """Maximum charge within a single hour"""
        return model.Ein[t] <= model.Rmax
    model.charge = Constraint(model.T, rule=charge_constraint) 
    
    def positive_discharge(model, t):
        'Limit discharge to the amount of charge in battery, including losses'
        return model.Eout[t] <= model.S[t]*np.sqrt(eff) / dih
    model.positive_discharge = Constraint(model.T, rule=positive_discharge)
    
    def positive_charge(model,t):
        return model.Ein[t] <= (model.Smax*dod-model.S[t])*np.sqrt(eff) / dih
    model.positive_charge = Constraint(model.T, rule=positive_charge)
        
    def binary_constraint_in(model,t):
        return model.Ein[t] <= model.X[t]*999999 
    model.binary_constraint1 = Constraint(model.T, rule=binary_constraint_in)
    
    def binary_constraint_out(model,t):
        return model.Eout[t] <= (1-model.X[t])*999999
    model.binary_constraint2 = Constraint(model.T, rule=binary_constraint_out)
    
    def soc_constraint(model,t):
        return model.S[t] <= model.Smax*dod
    model.soc = Constraint(model.T, rule=soc_constraint)
    
    
        
    real_expenses = capacity_val*model.Smax + power_val*model.Rmax + large_system_cost*model.alpha -  sum(generator_cost*dih*model.diss[t].value for t in model.T)

    #art_expenses = sum((model.Building_Power[t] - model.Solar[t]*model.alpha + model.Ein[t] - model.Eout[t]) for t in model.T)
    expenses = real_expenses#-art_expenses
    model.objective = Objective(expr=expenses, sense=minimize)

    # Solve the model
    solver = SolverFactory('gurobi')
    solver.solve(model)
    
    dataframe, capacity, power, alpha = microgrid_model_to_df(model)
    solar_cost = alpha*large_system_cost
    system_size = large_system_size*alpha
    return dataframe, capacity, power, solar_cost, alpha, system_size, model


    