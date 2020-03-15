import pandas as pd
from functools import reduce

def model_to_df(model,project_type):
    """
    Create a dataframe with hourly charge, discharge, state of charge, and
    output columns from a pyomo model.

    Parameters
    ----------
    model : pyomo model
        Model that has been solved

    Returns
    -------
    dataframe

    """
    # Need to increase the first & last data point by 1 because of pyomo indexing
    # and add 1 to the value of last model hour because of the range
    intervals = range(model.T.first(), model.T.last() + 1)
    Ein = [model.Ein[i].value for i in intervals]
    Eout = [model.Eout[i].value for i in intervals]
    charge_state = [model.S[i].value for i in intervals]
    if project_type == 'solar+storage':
        solar = [model.Solar[i] for i in intervals]
    else:
        solar = [0 for i in intervals]
    system_out = [(model.Output[i] - model.Eout[i].value + model.Ein[i].value) for i in intervals]
    df_dict = dict(
        intervals=intervals,
        solar=solar,
        Ein=Ein,
        Eout=Eout,
        charge_state=charge_state,
        system_out=system_out,
    )

    df = pd.DataFrame(df_dict)

    return df

def microgrid_model_to_df(model):
    """
    Create a dataframe with hourly charge, discharge, state of charge, and
    output columns from a pyomo model.

    Parameters
    ----------
    model : pyomo model
        Model that has been solved

    Returns
    -------
    dataframe

    """
    # Need to increase the first & last data point by 1 because of pyomo indexing
    # and add 1 to the value of last model hour because of the range
    intervals = range(model.T.first(), model.T.last() + 1)
    Ein = [model.Ein[i].value for i in intervals]
    Eout = [model.Eout[i].value for i in intervals]
    charge_state = [model.S[i].value for i in intervals]
    solar = [model.Solar[i] for i in intervals]
    capacity = model.Smax.value
    power = model.Rmax.value
    alpha = model.alpha.value

    df_dict = dict(
        intervals=intervals,
        solar=solar,
        Ein=Ein,
        Eout=Eout,
        charge_state=charge_state,

    )

    df = pd.DataFrame(df_dict)

    return df, capacity, power, alpha

# Demand Rate finder
def filter_demand_rates(df, num_columns=1):
    if num_columns >1:
        itr1 = ['demand_rate_category' + str(i) for i in range(1, num_columns+1)]
        itr2 = ['demand_rate' + str(i) for i in range(1, num_columns+1)]
        dics = {}
        cats = []
        for i1, i2 in zip(itr1, itr2):
            rate_categories = []
            dic = {}
            for rate_cat in set(df[i1]):
                rate_categories.append(rate_cat)
                dic[rate_cat] = df.loc[df[i1] == rate_cat, i2].values[0]
            dics.update(dic)
            cats.append(rate_categories)
        cats = reduce(lambda x,y: x+y,cats)        
        return cats, dics
    else:
        rate_categories = []
        dic = {}
        for rate_cat in set(df.demand_rate_category):
            rate_categories.append(rate_cat)
            dic[rate_cat] = df.loc[df.demand_rate_category == rate_cat, 'demand_rate'].values[0]
        return rate_categories, dic
