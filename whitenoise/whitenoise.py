#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 20:26:11 2019

@author: admin
"""

from random import gauss
from random import seed
import pandas as pd
from pandas import Series
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np 
import seaborn as sns 
import datetime 

# seed random number generator
seed(1)
# create white noise series
series = [gauss(0.0, 2.0) for i in range(1000)]
series = Series(series)

print(series.describe())

series.plot()
#histogram shows the tell-tale bell-curve shape.
series.hist()

#he correlogram does not show any obvious autocorrelation pattern.
autocorrelation_plot(series)


df = pd.read_csv('data.csv',index_col=[0], parse_dates=[0])
df.plot()
df.hist()
autocorrelation_plot(df)

def plot_df(df, x, y, title="", xlabel='Date', ylabel='Value', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

plot_df(df, x=df.index, y=df.y, title='daily withdrawals')    


df = pd.read_csv('data.csv', parse_dates=[0])
x = df['date']
y1 = df['y']

# Plot
fig, ax = plt.subplots(1, 1, figsize=(16,5), dpi= 120)
plt.fill_between(x, y1=y1, y2=-y1, alpha=0.5, linewidth=2, color='seagreen')
plt.ylim(-y1.max(), y1.max())
plt.title('daily withdrawals (Two Side View)', fontsize=16)
plt.hlines(y=0, xmin=np.min(df.date), xmax=np.max(df.date), linewidth=.5)
plt.show()

df = pd.read_csv('data.csv',index_col=[0], parse_dates=[0])
df.reset_index(inplace=True)

df['year'] = [d.year for d in df.date]
df['month'] = [d.strftime('%b') for d in df.date]
years = df['year'].unique()

df_monthly_avg = df.groupby(['year','month'])['y'].mean().reset_index()

df_monthly_avg.to_csv('monthly.csv', index=False)
d = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
df_monthly_avg.month = df_monthly_avg.month.map(d)

df_monthly_avg['y'].plot()
df_monthly_avg['y'].hist()
autocorrelation_plot(df_monthly_avg['y'])

x = df_monthly_avg['month']
y1 = df_monthly_avg['y']

df = df_monthly_avg.copy(deep=True)

# Prep Colors
np.random.seed(100)
mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(years), replace=False)

# Draw Plot
plt.figure(figsize=(16,12), dpi= 80)
for i, y in enumerate(years):
    if i > 0:        
        plt.plot('month', 'y', data=df.loc[df.year==y, :].sort_values(by=['month']), color=mycolors[i], label=y)
        plt.text(df.loc[df.year==y, :].shape[0]-.9, df.loc[df.year==y, 'y'][-1:].values[0], y, fontsize=12, color=mycolors[i])


# Draw Plot
fig, axes = plt.subplots(1, 2, figsize=(20,7), dpi= 80)
sns.boxplot(x='year', y='y', data=df, ax=axes[0])
sns.boxplot(x='month', y='y', data=df.loc[~df.year.isin([2015, 2019]), :])

# Set Title
axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=18); 
axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=18)
plt.show()

d = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}

 
df = pd.read_csv('monthly.csv',index_col=[0], parse_dates=[0])
df.reset_index(inplace=True)
df.month = df.month.map(d)

df['date'] = pd.to_datetime(df['year'])
df['year'] = pd.DatetimeIndex(df['date']).year
df['date'] = pd.DatetimeIndex(df['year']).year + df['month'] 
df['day'] =1
df['date'] = df.year.astype(str) + '-' + df.month.astype(str) + '-' + df.day.astype(str) 
df['date'] = pd.to_datetime(df['date'])

df.drop(['year', 'month','day'], axis=1, inplace=True) 
df.set_index('date',inplace=True)


fig, axes = plt.subplots(1,3, figsize=(20,4), dpi=100)
df.plot(title='Trend Only', legend=False, ax=axes[0])
df.plot(title='Seasonality Only', legend=False, ax=axes[1])
df.plot(title='Trend and Seasonality', legend=False, ax=axes[2])


fig, axes = plt.subplots(1,3, figsize=(20,4), dpi=100)
pd.read_csv('guinearice.csv', parse_dates=['date'], index_col='date').plot(title='Trend Only', legend=False, ax=axes[0])

pd.read_csv('sunspotarea.csv', parse_dates=['date'], index_col='date').plot(title='Seasonality Only', legend=False, ax=axes[1])

pd.read_csv('AirPassengers.csv', parse_dates=['date'], index_col='date').plot(title='Trend and Seasonality', legend=False, ax=axes[2])
