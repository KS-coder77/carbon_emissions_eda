#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


# In[1]:


data = pd.read_csv(r"C:\Users\Krupa\Documents\Krups Coding\Kaggle\DA_CO2emissions\co2emissions.csv")
data.head()


# In[3]:


data.info()


# In[5]:


data.shape


# In[6]:


data.describe()


# In[9]:


def check_data(df):
    summary = [
        [col, df[col].dtype, df[col].count(), df[col].nunique(), df[col].isnull().sum(), df.duplicated().sum()]
        for col in df.columns] 
    
    df_check = pd.DataFrame(summary, columns = ['column', 'dtype', 'instances', 'unique', 'missing_vals', 'duplicates'])
    
    return df_check 


# In[10]:


check_data(data)


# In[86]:


# change date col from int to datetime
def convert_to_datetime(df, column_name):
    try: 
        df[column_name] = pd.to_datetime(df[column_name], format = "%Y")
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None 


# In[87]:


convert_to_datetime(data, 'Year')


# In[ ]:





# In[11]:


print('Missing values (%) per field:\n', 100*data.isnull().mean())


# A large percentage of values are missing from this dataset, let's assume these values are zero. 

# In[12]:


data.fillna(0, inplace=True)


# In[13]:


data.head()


# In[14]:


print('Missing values (%) per field:\n', 100*data.isnull().mean())


# In[16]:


#let's take a closer look at the ISO column 

data.columns


# In[17]:


data['ISO 3166-1 alpha-3'].unique()


# In[19]:


data['ISO 3166-1 alpha-3'].nunique()


# In[20]:


data['Country'].unique()


# In[21]:


data['Country'].nunique()


# The ratio of countries to ISO is unbalanced. There are 6 more countries than ISO's. This suggests some ISO's may be used multiple times, which makes sense as some Islands form part of countries.
# 

# In[23]:


data.loc[data['Country'] == 'International Transport']


# In[24]:


data.loc[data['Country'] == 'Global']


# It's unclear what global and internatioal transport refer to. For now, let's remove these rows from our analysis, and consider them separately.

# In[38]:


world_data = data[(data['Country']=='Global') | (data['Country']=='International Transport')]
world_data.head()
#filtered_rows = df[(df['Category'] == 'A') & (df['Status'] == 'X')]

#df_filtered = df[df['A'] <= 50] #(removes rows where vals greater than 50)


# In[39]:


world_data.shape


# In[89]:


#let's consider only the data from actual countries and from 2000 onwrds
filtered_data = data[(data['Country']!='Global') & (data['Country']!='International Transport') & (data['Year']>='2000')]
filtered_data.head()


# In[90]:


filtered_data.shape


# In[91]:


filtered_data['Country']


# In[ ]:





# In[ ]:





# In[ ]:


# questions to ask ourselves 

# 1. how have the emissions changed over time for coal, oil, gas, cement, flaring, other 
# 2. what can we forecast for the next 5-10 years? 
# 3. who are the top contributors to emissions for coal, oil, gas, cement, flaring,other (overall)
# 4. are there any correlations?
# 5. plot total avg. emissions by top 10 countries 
# 6. global plots chloropleth? 



# In[116]:


countries = list(filtered_data.drop(columns=['Year']).groupby('Country').sum().sort_values(by='Total',ascending=False).index)
values = list(filtered_data.drop(columns=['Year']).groupby('Country').sum().sort_values(by='Total',ascending=False)['Total'])

plt.figure(figsize=(12,5))
plt.title(label='Top 10 Countries Total Emissions, 2000 - 2021')
sns.set_style('darkgrid')
sns.barplot(x=countries[:10], y=values[:10], hue = countries[:10], palette='Set2',edgecolor='.2', legend='auto')


# In[93]:


#let's take a closer look at China over time 
china_data = filtered_data[filtered_data['Country']=='China']
china_data


# In[94]:


plt.figure(figsize=(12,5))
plt.subplot(121)
sns.lineplot(x='Year', y ='Total', data=china_data)
plt.title(label='China Total Emissions Trend, 2000 - 2021')

plt.subplot(122)
sns.barplot(x='Year', y='Total', data=china_data, hue='Year', palette='Set3', edgecolor='.3', legend=False)
plt.xticks(rotation=45)
plt.tight_layout()
plt.title(label='China Total Emissions Trend, 2000 - 2021')


# In[95]:


filtered_data.columns


# In[110]:


#let's consider the average annual emissions for all categories
filtered_df = filtered_data.copy()
filtered_df.set_index('Year', inplace=True)


# In[120]:


annual_avg =filtered_df[['Total', 'Coal', 'Oil', 'Gas', 'Cement', 'Flaring', 'Other']].resample('Y').mean().round(2)
annual_avg


# In[109]:


sns.lineplot(data=annual_avg)
plt.title(label='Annual Average Emissions, 2000 - 2021')


# The above graph indicates that coal, followed by oil and gas are on average the top emittors of carbon.Between 2020 and 2021 there is a slight dip due to the pandemic. Other than that, there seems to be a steady increase across the board. 

# In[136]:


sns.regplot(data=annual_avg, x='Coal', y='Oil')
plt.title(label='Relationship between Coal and Oil emissions')


# In[137]:


sns.regplot(data=annual_avg, x='Coal', y='Gas')
plt.title(label='Relationship between Coal and Gas emissions')


# In[138]:


sns.regplot(data=annual_avg, x='Coal', y='Cement')
plt.title(label='Relationship between Coal and Cement emissions')


# In[139]:


sns.regplot(data=annual_avg, x='Gas', y='Cement')
plt.title(label='Relationship between Gas and Cement emissions')


# In[140]:


sns.regplot(data=annual_avg, x='Flaring', y='Cement')
plt.title(label='Relationship between Flaring and Cement emissions')


# In[118]:


plt.figure(figsize=(12,6))
sns.heatmap(annual_avg.corr(), cmap='crest', annot=True)
plt.show()


# In[124]:


percapita_df = filtered_df.groupby('Country')[['Per Capita']].mean().sort_values('Per Capita', ascending=False).head(10).reset_index()
percapita_df


# In[133]:


sns.barplot(percapita_df, x ='Country', y='Per Capita', hue='Country')
plt.xticks(rotation=70)
plt.tight_layout()
plt.title(label='Top 10 Countries by Total Emissions Per Capita, 2000 - 2021')


# In[ ]:





# In[22]:


smalldf = data[['Country', 'ISO 3166-1 alpha-3']]
smalldf

