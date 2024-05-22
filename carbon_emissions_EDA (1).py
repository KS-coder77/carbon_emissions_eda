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


# The data is organized using the following columns: 
# - Country (the name of the country)
# - ISO 3166-1 alpha-3 (the three letter code for the country)
# - Year (the year of survey data)
# - Total (the total amount of CO2 emitted by the country in that year)
# - Coal (amount of CO2 emitted by coal in that year)
# - Oil (amount emitted by oil) 
# - Gas (amount emitted by gas) 
# - Cement (amount emitted by cement)
# - Flaring (flaring emission levels ) 
# - Other (other forms such as industrial processes )
# - Per Capita which provides an insight into how much personal carbon dioxide emission is present in each Country per individual 
# 

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


# questions to ask ourselves 

# 1. how have the emissions changed over time for coal, oil, gas, cement, flaring, other 
# 2. what can we forecast for the next 5-10 years? 
# 3. who are the top contributors to emissions for coal, oil, gas, cement, flaring,other (overall)
# 4. are there any correlations?
# 5. plot total avg. emissions by top 10 countries 
# 6. global plots chloropleth? 


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


# In[39]:


world_data.shape


# ## Filtered Data

# In[89]:


#let's consider only the data from actual countries and from 2000 onwrds
filtered_data = data[(data['Country']!='Global') & (data['Country']!='International Transport') & (data['Year']>='2000')]
filtered_data.head()


# In[90]:


filtered_data.shape


# In[91]:


filtered_data['Country']


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


# In[181]:


annual_avg['% Coal'] = (annual_avg['Coal']/annual_avg['Total'])*100
annual_avg['% Oil'] = (annual_avg['Oil']/annual_avg['Total'])*100
annual_avg['% Gas'] = (annual_avg['Gas']/annual_avg['Total'])*100
annual_avg['% Cement'] = (annual_avg['Cement']/annual_avg['Total'])*100
annual_avg['% Flaring'] = (annual_avg['Flaring']/annual_avg['Total'])*100
annual_avg['% Other'] = (annual_avg['Other']/annual_avg['Total'])*100


# In[185]:


annual_avg = annual_avg.round(2)
annual_avg


# In[191]:


annual_avg.loc[annual_avg['% Oil'] > annual_avg['% Coal']]


# In[196]:


annual_avg.loc[annual_avg['% Gas'] > annual_avg['% Oil']]


# There appears to only be two years when Oil carbon emissions were higher than Coal carbon emissions, and that was in 2000 and 2001.

# In[109]:


sns.lineplot(data=annual_avg)
plt.title(label='Annual Average Emissions, 2000 - 2021')


# The above graph indicates that coal, followed by oil and gas are on average the top emittors of carbon.Between 2020 and 2021 there is a slight dip due to the pandemic. Other than that, there seems to be a steady increase across the board. 

# In[143]:


full_countries_data = data[(data['Country']!='Global') & (data['Country']!='International Transport')]
full_countries_data.shape


# In[148]:


full_countries_data.head()


# In[160]:


full_countries_data['Country'].nunique()


# In[144]:


full_countries_df = full_countries_data.copy()
full_countries_df.set_index('Year', inplace=True)
annual_avg_full_data = full_countries_df[['Total', 'Coal', 'Oil', 'Gas', 'Cement', 'Flaring', 'Other']].resample('Y').mean().round(2)


# In[ ]:





# In[147]:


plt.figure(figsize=(12,8))
sns.lineplot(data=annual_avg_full_data)
plt.title(label='Annual Average Emissions')


# In[156]:


df_1920 = full_countries_data[full_countries_data['Year']>='1920-01-01']
df_1920


# In[165]:


plt.figure(figsize=(12,8))
plt.title(label='Per Capita Emissions Trend, 1920 - 2021')
sns.lineplot(data=full_countries_data[full_countries_data['Year']>='1920-01-01'], x='Year', y='Per Capita')


# In[164]:


plt.figure(figsize=(12,8))
plt.title(label='Total Emissions Trend, 1920 - 2021')
sns.lineplot(data=full_countries_data[full_countries_data['Year']>='1920-01-01'], x='Year', y='Total')


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


# What's interesting to notice here, is that the Total has a close correlation with Cement followed by Coal. Given that the overall emissions from Cement are at the lower end compared to the other fossil fuels, it's surprising to see such a close correlation with the Total.

# In[182]:


sns.regplot(data=annual_avg, x='Cement', y='Total')
plt.title(label='Relationship between Cement and Total emissions')


# In[183]:


sns.regplot(data=annual_avg, x='Coal', y='Total')
plt.title(label='Relationship between Coal and Total emissions')


# In[124]:


percapita_df = filtered_df.groupby('Country')[['Per Capita']].mean().sort_values('Per Capita', ascending=False).head(10).reset_index()
percapita_df


# In[133]:


sns.barplot(percapita_df, x ='Country', y='Per Capita', hue='Country')
plt.xticks(rotation=70)
plt.tight_layout()
plt.title(label='Top 10 Countries by Total Emissions Per Capita, 2000 - 2021')


# ## Feature Engineering 

# In order for a feature to be useful, it must have a relationship to the target that your model is able to learn (e.g. Linear models are only able to learn linear relationships). Hence, when using a linear model our goal is to transform the features to make their relationship to the target linear. 
# 
# To begin, let's consider the feature utility metric - mutual information (MI) which measures the relationship between a potential feature and the target. 

# In[205]:


a.head()


# In[241]:


X = a.copy()
y = X.pop('Total')


# Scikit-learn has two MI metrics. We will use mutual_info_regression as it's suited to real-valued targets like our Total emissions column.

# In[242]:


X.dtypes


# In[243]:


#extract year only from datetime column 
X['year'] = X['Year'].dt.year


# In[244]:


X = X.drop(columns=['Year'])


# In[245]:


X.columns


# In[246]:


#convert to int data type
X[['Coal', 'Oil', 'Gas', 'Cement', 'Flaring', 'Other', '% Coal',
       '% Oil', '% Gas', '% Cement', '% Flaring', '% Other', 'year']] = X[['Coal', 'Oil', 'Gas', 'Cement', 'Flaring', 'Other', '% Coal',
       '% Oil', '% Gas', '% Cement', '% Flaring', '% Other', 'year']].astype(int)


# In[247]:


X.dtypes


# In[248]:


y


# In[249]:


discrete_features = X.dtypes == int


# In[250]:


X


# In[253]:


y = y.astype(int)


# In[254]:


y.dtypes


# In[255]:


discrete_features


# In[256]:


from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name='MI Scores', index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X, y, discrete_features)
mi_scores[::3]


# In[22]:


smalldf = data[['Country', 'ISO 3166-1 alpha-3']]
smalldf

