#%%
#Importing modules 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point, Polygon



#%%
#Importing datasets 

crime = pd.read_csv('Crime_Incidents_in_2022.csv')
stop = pd.read_csv('Stop_Data.csv')


# %%
#Selecting columns of interest 

#X long 
#y lat 

crimedf = crime[['X',
                 'Y', 
                 'SHIFT', 
                 'METHOD', 
                 'OFFENSE', 
                 'BLOCK',
                 'DISTRICT',
                 'START_DATE']]

stopdf = stop[['GENDER', 
               'ETHNICITY', 
               'AGE', 
               'INVESTIGATIVE', 
               'DATETIME']]


#%% 
#Cleaning the data set 

# Checking for Null values
crimedf.isna().sum()

#Dropping null values
crimedf.dropna(axis=0,inplace=True)

#Converting START_DATE to timestamp
crimedf['START_DATE'] = pd.to_datetime(crimedf['START_DATE'])

#Creating new column for hour
crimedf["Hour"]=crimedf["START_DATE"].dt.hour

#Creating new column for month
crimedf["Month"]=crimedf["START_DATE"].dt.month

#Creating new column for dayofweek 
crimedf["DayofWeek"]=crimedf["START_DATE"].dt.day_name()


#%%
#EDA

# Crimdedf: barchart of frequency by district, barchart of frequency by month, barchart of frequency by hour

#Countplot for hour
sns.countplot(x ='Hour', data = crimedf,palette = "Set2").set(title='Countplot for frequncy of crime during differet hours')

#Countplot for month
sns.countplot(x ='Month', data = crimedf,palette = "Set2").set(title='Countplot for frequncy of crime during differet months')

#Countplot for day of week
sns.countplot(x ='DayofWeek', data = crimedf,palette = "Set2").set(title='Countplot for frequncy of crime during days of week')

#pie chart for day of week
plt.pie(crimedf['DayofWeek'].value_counts().values, labels=crimedf['DayofWeek'].value_counts().index, colors=sns.color_palette('colorblind'), autopct='%.2f')
plt.title("Pie chart showing crime during different days of the week",bbox={'facecolor':'1.0', 'pad':5})
plt.legend(bbox_to_anchor=(2,0.5), loc="center right")
plt.show()

#Countplot for district
sns.countplot(x ='DISTRICT', data = crimedf,palette = "Set2").set(title='Countplot for frequncy of crime in different districts')

#%%

street_map = gpd.read_file('/Users/tylerwallett/Downloads/Police_Districts')

street_map.plot()

#%%
crs = {'init': 'epsg:4326'}

geometry = [Point(xy) for xy in zip( crimedf['X'], crimedf['Y'])]

geo_df = gpd.GeoDataFrame(crimedf,
                 crs=crs,
                 geometry= geometry)

#%%

fig, ax = plt.subplots(figsize = (15,15))

street_map.plot(ax =ax)

geo_df[geo_df["OFFENSE"] == 'HOMICIDE'].plot(ax=ax, color = 'red')

#%%



#%%
#Modeling 



#Conclusions 