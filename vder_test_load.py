# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:50:53 2019

@author: Richard_Fu
"""

import helper
df = helper.load_data('vder_test')

#choose 10 days to be
#june 21 5-7pm
#june 24 4-6pm
#july 8 2-4pm
#july 13 3-5pm
#july 24 5-7pm
#august 1 3-5pm
#august 9 5-7pm
#august 13 5-7pm
#august 25 2-4pm
#september 5 4-6pm

test_dates = df.loc[(df.date.dt.month==6)&(df.date.dt.day.isin([21,22,23,24]))]