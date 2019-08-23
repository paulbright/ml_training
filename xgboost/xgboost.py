#!/usr/bin/env python
# coding: utf-8

# # Hourly Time Series Forecasting using XGBoost
# 
# [If you haven't already first check out my previous notebook forecasting on the same data using Prophet](https://www.kaggle.com/robikscube/hourly-time-series-forecasting-with-prophet)
# 
# In this notebook we will walk through time series forecasting using XGBoost. The data we will be using is hourly energy consumption.

# In[ ]:

#import sys
#print(sys.base_prefix)
#
#
#import pip
#from pip._internal import main
#
#main(['install', 'xgboost'])

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.style.use('fivethirtyeight')


# # Data
# The data we will be using is hourly power consumption data from PJM. Energy consumtion has some unique charachteristics. It will be interesting to see how prophet picks them up.
# 
# Pulling the `PJM East` which has data from 2002-2018 for the entire east region.

# In[ ]:


df = pd.read_csv('processed/1053_OR_TAMBO_INTERNATIONAL_AIRPORT.csv', index_col=[1], parse_dates=[1])

df = df.drop(['CASH_OUT','year', 'month', 'day', 'week', 'weekday', 'holiday',
       'daystoholiday'], axis=1)
# In[ ]:


color_pal = ["#F8766D", "#D39200", "#93AA00", "#00BA38", "#00C19F", "#00B9E3", "#619CFF", "#DB72FB"]
_ = df.plot(style='.', figsize=(15,5), color=color_pal[0], title='ORT')


# # Train/Test Split
# Cut off the data after 2015 to use as our validation set.

# In[ ]:


split_date = '2018-01-03'
df_train = df.loc[df.index <= split_date].copy()
df_test = df.loc[df.index > split_date].copy()


# In[ ]:


_ = df_test.rename(columns={'actual_cash_out': 'TEST SET'}).join(df_train.rename(columns={'actual_cash_out': 'TRAINING SET'}), how='outer').plot(figsize=(15,5), title='ORT', style='.')


# # Create Time Series Features

# In[ ]:


def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    #df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    
    X = df[['dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X


# In[ ]:


X_train, y_train = create_features(df_train, label='actual_cash_out')
X_test, y_test = create_features(df_test, label='actual_cash_out')


# # Create XGBoost Model

# In[ ]:


reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50,
       verbose=False) # Change verbose to True if you want to see it train


# ## Feature Importances
# Feature importance is a great way to get a general idea about which features the model is relying on most to make the prediction. This is a metric that simply sums up how many times each feature is split on.
# 
# We can see that the day of year was most commonly used to split trees, while hour and year came in next. Quarter has low importance due to the fact that it could be created by different dayofyear splits.

# In[ ]:


_ = plot_importance(reg, height=0.9)


# # Forecast on Test Set

# In[ ]:


df_test['cash_out_Prediction'] = reg.predict(X_test)
df_all = pd.concat([df_test, df_train], sort=False)


# In[ ]:


_ = df_all[['actual_cash_out','cash_out_Prediction']].plot(figsize=(15, 5))


# # Look at first month of predictions

# In[ ]:


# Plot the forecast with the actuals
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
_ = df_all[['cash_out_Prediction','actual_cash_out']].plot(ax=ax,
                                              style=['-','.'])
ax.set_xbound(lower='2018-02-01', upper='2019-01-01')
ax.set_ylim(0, 350000)
plot = plt.suptitle('March 2018 Forecast vs Actuals')


# 

# In[ ]:


# Plot the forecast with the actuals
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
_ = df_all[['cash_out_Prediction','actual_cash_out']].plot(ax=ax,
                                              style=['-','.'])
ax.set_xbound(lower='2018-03-01', upper='2018-03-07')
ax.set_ylim(0, 350000)
plot = plt.suptitle('First Week of March Forecast vs Actuals')


# In[ ]:


f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
_ = df_all[['cash_out_Prediction','actual_cash_out']].plot(ax=ax,
                                              style=['-','.'])
ax.set_ylim(0, 350000)
ax.set_xbound(lower='2018-07-01', upper='2018-07-08')
plot = plt.suptitle('First Week of July Forecast vs Actuals')


# # Error Metrics On Test Set
# Our RMSE error is 13780445  
# Our MAE error is 2848.89  
# Our MAPE error is 8.9%

# In[ ]:


mse = mean_squared_error(y_true=df_test['actual_cash_out'],
                   y_pred=df_test['cash_out_Prediction'])

import math

rmse = math.sqrt(mse)

print('rmse:', rmse)

# In[ ]:


mae = mean_absolute_error(y_true=df_test['actual_cash_out'],
                   y_pred=df_test['cash_out_Prediction'])

print('mae:', mae)
# I like using mean absolute percent error because it gives an easy to interperate percentage showing how off the predictions are.
# MAPE isn't included in sklearn so we need to use a custom function.

# In[ ]:


def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[ ]:


mean_absolute_percentage_error(y_true=df_test['actual_cash_out'],
                   y_pred=df_test['cash_out_Prediction'])


# # Look at Worst and Best Predicted Days

# In[ ]:


df_test['error'] = df_test['actual_cash_out'] - df_test['cash_out_Prediction']
df_test['abs_error'] = df_test['error'].apply(np.abs)
error_by_day = df_test.groupby(['year','month','dayofmonth']).mean()[['actual_cash_out','cash_out_Prediction','error','abs_error']]


# In[ ]:


# Over forecasted days
error_by_day.sort_values('error', ascending=True).head(10)


# Notice anything about the over forecasted days? 
# - #1 worst day - July 4th, 2016 - is a holiday. 
# - #3 worst day - December 25, 2015 - Christmas
# - #5 worst day - July 4th, 2016 - is a holiday.   
# Looks like our model may benefit from adding a holiday indicator.

# In[ ]:


# Worst absolute predicted days
error_by_day.sort_values('abs_error', ascending=False).head(10)


# The best predicted days seem to be a lot of october (not many holidays and mild weather) Also early may

# In[ ]:


# Best predicted days
error_by_day.sort_values('abs_error', ascending=True).head(10)


# # Plotting some best/worst predicted days

# In[ ]:


f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(10)
_ = df_all[['cash_out_Prediction','actual_cash_out']].plot(ax=ax,
                                              style=['-','.'])
ax.set_ylim(0, 350000)
ax.set_xbound(lower='2018-08-01', upper='2018-09-01')
plot = plt.suptitle('Aug , 2018 - Worst Predicted Day')


# This one is pretty impressive. SPOT ON!

# In[ ]:


f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(10)
_ = df_all[['cash_out_Prediction','actual_cash_out']].plot(ax=ax,
                                              style=['-','.'])
ax.set_ylim(0, 350000)
ax.set_xbound(lower='2018-09-01', upper='2018-10-01')
plot = plt.suptitle('Oct 2018 - Best Predicted Day')


# In[ ]:


f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(10)
_ = df_all[['cash_out_Prediction','actual_cash_out']].plot(ax=ax,
                                              style=['-','.'])
ax.set_ylim(0, 350000)
ax.set_xbound(lower='2018-08-13', upper='2018-08-30')
plot = plt.suptitle('Aug 13, 2018 - Worst Predicted Day')


# # Up next?
# - Add Lag variables
# - Add holiday indicators.
# - Add weather data source.
