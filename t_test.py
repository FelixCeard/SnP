#!/usr/bin/env python
# coding: utf-8

# In[101]:


import pandas as pd
import glob
import datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t


# 

# In[129]:


# Read all data
files = glob.glob(r'C:\Users\christina\PycharmProjects\SnP\data\pyfinancialdata\data\stocks\histdata\GRXEUR\*.csv')
files


# In[132]:


# read first file
file_path = files.pop(0)
df_file = pd.read_csv(file_path, sep=';', header=None)
df_file[0] = pd.to_datetime(df_file[0])
df_file['minute_average'] = (df_file[1] + df_file[4])/2

for file_path in files:
    _df_file = pd.read_csv(file_path, sep=';', header=None)
    _df_file[0] = pd.to_datetime(_df_file[0])
    _df_file['minute_average'] = (_df_file[1] + _df_file[4])/2
    
    df_file = pd.concat([df_file, _df_file])
    
df_file


# In[26]:


# 2010
path_2010 = r'C:\Users\christina\PycharmProjects\SnP\data\pyfinancialdata\data\stocks\histdata\SPXUSD\DAT_ASCII_SPXUSD_M1_2010.csv'
df_2010 = pd.read_csv(path_2010, sep=';', header=None)
# df_2010.index.name = 'date'
df_2010[0] = pd.to_datetime(df_2010[0])
df_2010['minute_average'] = (df_2010[1] + df_2010[4])/2
df_2010.sort_index(inplace=True, axis=0)
df_2010


# In[133]:


df_file.iloc[0][0], df_file.iloc[-1][0]


# In[134]:


df_2010 = df_file


# In[135]:


# 20101114 180300
days = []
time_start = df_2010.iloc[0][0]

current_time = time_start
next_time = time_start + datetime.timedelta(hours=24)

prices = []
current_index = 0
for i in tqdm(range(len(df_2010)), total=len(df_2010)):
    if df_2010.iloc[i][0] >= next_time:
        # days.append(prices)
        days.append(df_2010.iloc[current_index:i])
        # prices = []
        next_time = next_time + datetime.timedelta(hours=24)
        current_index = i
    # else:
        # prices.append(df_2010.iloc[i]['minute_average'].item())   
    


# In[146]:


# create the cutpoints
means = []
for day_data in tqdm(days, total=len(days)):
    # print(day_data.head())
    time_start = day_data.iloc[0][0]
    
    # print(time_start)
    
    # Get the date of the next day
    next_day = time_start + datetime.timedelta(days=1)
    next_day_date = next_day.date()
    
    # Combine the date of the next day with the time 12:00
    next_day_12 = datetime.datetime.combine(time_start.date(), datetime.time(minute=30)) + datetime.timedelta(hours=13)
    # print(next_day_12)
    
    # only keep the data after 12:00
    data_after_12 = day_data[day_data[0] > next_day_12]
    
    # data after 14:30
    next_day_14 = datetime.datetime.combine(time_start.date(), datetime.time(minute=30)) + datetime.timedelta(hours=14)
    data_after_12 = data_after_12[data_after_12[0] < next_day_14]
    data_after_14_30 = day_data[day_data[0] > next_day_14]

    next_day_15 = datetime.datetime.combine(time_start.date(), datetime.time(minute=30)) + datetime.timedelta(hours=15)
    data_after_14_30 = data_after_14_30[data_after_14_30[0] < next_day_15]
    data_after_15_30 = day_data[day_data[0] > next_day_15]
    
    next_day_16 = datetime.datetime.combine(time_start.date(), datetime.time(minute=30)) + datetime.timedelta(hours=16)
    data_after_15_30 = data_after_15_30[data_after_15_30[0] < next_day_16]
    
    average_before = data_after_12['minute_average'].mean()
    average_14 = data_after_14_30['minute_average'].mean()
    average_15 = data_after_15_30['minute_average'].mean()
    
    # print(average_before, average_14, average_15)
    # break
    if float('nan') not in [average_before, average_14, average_15] and np.nan not in [average_before, average_14, average_15]:
        means.append((average_before, average_14, average_15))


# In[ ]:





# In[148]:


# compute difference in means
df = pd.DataFrame(means)
df.columns = ['12', '14', '15']

df['delta_14'] = df['14'] - df['12']
df['delta_15'] = df['15'] - df['14']
df['delta_15_12'] = df['15'] - df['12']


# In[155]:


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# sns.lineplot(df, x=df.index, y='12', ax=ax, label='12')
# sns.lineplot(df, x=df.index, y='14', ax=ax, label='14')
# sns.lineplot(df, x=df.index, y='15', ax=ax, label='15')
sns.lineplot(df, x=df.index, y='delta_14', ax=ax, label='14 to 12')
sns.lineplot(df, x=df.index, y='delta_15', ax=ax, label='15 to 14')
sns.lineplot(df, x=df.index, y='delta_15_12', ax=ax, label='15 to 12')


# In[156]:


# 
df['average_post_intervention'] = df['12'] - 0.5*(df['14'] + df['15'])
df['positive'] = False
df['positive'][df['average_post_intervention'] >= 0] = True

df


# In[157]:


df_positive = df[df['positive'] == True]
df_negative = df[df['positive'] == False]


# H_0: μ_X − μ_Y = 0

# ## Negative

# In[168]:


print('relative values - Negative News')
# estimating global variance
# global_variance = df_negative[['12', '14', '15']].values.reshape(-1).var()
global_variance = df_negative[['delta_14']].values.reshape(-1).var()
global_variance2 = df_negative[['delta_15']].values.reshape(-1).var()
global_variance3 = df_negative[['delta_15_12']].values.reshape(-1).var()
# global_expectation = df_negative[['12', '14', '15']].values.reshape(-1).mean()
dof = 2*len(df_negative) - 2
print('degree of Freedom:', dof)

mean_12 = df_negative['delta_14'].mean()
mean_14 = df_negative['delta_15'].mean()
mean_15 = df_negative['delta_15_12'].mean()
# mean_15 = df_negative['15'].mean()

t = (mean_12 - 0) / (global_variance * np.sqrt(len(df_negative)))
print('')
print('Mu 𝞓(14:30-13:30) = 0')
print('t:', t)
t = (mean_14 - 0) / (global_variance2 * np.sqrt(len(df_negative)))
print('')
print('Mu 𝞓(15:30-14:30) = 0')
print('t:', t)
print('')
t = (mean_15 - 0) / (global_variance3 * np.sqrt(len(df_negative)))
print('Mu 𝞓(15:30-13:30) = 0')
print('t:', t)
print('')


# In[172]:


# estimating global variance
global_variance = df_negative[['12', '14', '15']].values.reshape(-1).var()
# global_variance = df_negative[['delta_14', 'delta_15']].values.reshape(-1).var()
# global_expectation = df_negative[['12', '14', '15']].values.reshape(-1).mean()
dof = 2*len(df_negative) - 2
print('dof:', dof)

mean_12 = df_negative['12'].mean()
mean_14 = df_negative['14'].mean()
mean_15 = df_negative['15'].mean()

t_14_12 = (mean_12 - mean_14 - 0) / (global_variance * (2/len(df_negative)))
print('Mu 14:30 = Mu 13:30')
print('t =', t_14_12)
# print('p_value:', 2*(1 - t.cdf(abs(t_14_12), dof)))
print('')

t_15_12 = (mean_12 - mean_15 - 0) / (global_variance * (2/len(df_negative)))
print('Mu 15:30 = Mu 13:30')
print('t =', t_15_12)
# print('p_value:', 2*(1 - t.cdf(abs(t_15_12), dof)))
print('')

t_15_14 = (mean_14 - mean_15 - 0) / (global_variance * (2/len(df_negative)))
print('Mu 15:30 = Mu 14:30')
print('t =', t_15_14)
# print('p_value:', 2*(1 - t.cdf(abs(t_15_14), dof)))
print('')


# ## Positive

# In[173]:


print('relative values - Positive News')
# estimating global variance
# global_variance = df_negative[['12', '14', '15']].values.reshape(-1).var()
global_variance = df_positive[['delta_14']].values.reshape(-1).var()
global_variance2 = df_positive[['delta_15']].values.reshape(-1).var()
global_variance3 = df_positive[['delta_15_12']].values.reshape(-1).var()
# global_expectation = df_negative[['12', '14', '15']].values.reshape(-1).mean()
dof = 2*len(df_positive) - 2
print('degree of Freedom:', dof)

mean_12 = df_positive['delta_14'].mean()
mean_14 = df_positive['delta_15'].mean()
mean_15 = df_positive['delta_15_12'].mean()
# mean_15 = df_negative['15'].mean()

t = (mean_12 - 0) / (global_variance * np.sqrt(len(df_positive)))
print('')
print('Mu 𝞓(14:30-13:30) = 0')
print('t:', t)
t = (mean_14 - 0) / (global_variance2 * np.sqrt(len(df_positive)))
print('')
print('Mu 𝞓(15:30-14:30) = 0')
print('t:', t)
print('')
t = (mean_15 - 0) / (global_variance3 * np.sqrt(len(df_positive)))
print('Mu 𝞓(15:30-13:30) = 0')
print('t:', t)
print('')


# In[174]:


# estimating global variance
global_variance = df_positive[['12', '14', '15']].values.reshape(-1).var()
# global_expectation = df_negative[['12', '14', '15']].values.reshape(-1).mean()
dof = 2*len(df_positive) - 2
print('dof:', dof)

mean_12 = df_positive['12'].mean()
mean_14 = df_positive['14'].mean()
mean_15 = df_positive['15'].mean()

t_14_12 = (mean_12 - mean_14 - 0) / (global_variance * (2/len(df_positive)))
print('Mu 14:30 = Mu 13:30')
print('t =', t_14_12)
# print('p_value:', 2*(1 - t.cdf(abs(t_14_12), dof)))
print('')

t_15_12 = (mean_12 - mean_15 - 0) / (global_variance * (2/len(df_positive)))
print('Mu 15:30 = Mu 13:30')
print('t =', t_15_12)
# print('p_value:', 2*(1 - t.cdf(abs(t_15_12), dof)))
print('')

t_15_14 = (mean_14 - mean_15 - 0) / (global_variance * (2/len(df_positive)))
print('Mu 15:30 = Mu 14:30')
print('t =', t_15_14)
# print('p_value:', 2*(1 - t.cdf(abs(t_15_14), dof)))
print('')

