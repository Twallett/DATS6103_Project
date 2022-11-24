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
#Reformatting 

#Reformatting Shift: 1- Midnight, 2- Evening and 3- Day

crimedf['SHIFT'] = crimedf['SHIFT'].map(lambda x: 1 if x == 'MIDNIGHT' else 2 if x == 'EVENING' else 3 if x == 'DAY' else np.nan)

#%%
#Reformatting Method: 1- Knife, 2- Gun and 3 - Others

crimedf['METHOD'] = crimedf['METHOD'].map(lambda x: 1 if x == 'KNIFE' else 2 if x == 'GUN' else 3 if x == 'OTHERS' else np.nan)

#%%
#Reformatting Offense: 1- Homicide, 2-Theft, 3- Motor vehicle theft, 4- Assault, 5- Robbery, 6- Burglary, 7- Sex abuse and 8- Arson

crimedf['OFFENSE'] = crimedf['OFFENSE'].map(lambda x: 1 if x == 'HOMICIDE' else 2 if x == 'THEFT/OTHER' else 3 if (x == 'MOTOR VEHICLE THEFT')|(x == 'THEFT F/AUTO') else 4 if x == 'ASSAULT W/DANGEROUS WEAPON' else 5 if x == 'ROBBERY' else 6 if x == 'BURGLARY' else 7 if x == 'SEX ABUSE' else 8 if x == 'ARSON' else np.nan)

#%%
#Reformatting of Day of Week: 1- Monday, 2-Tuesday, and so on...

crimedf['NewDayofWeek'] = crimedf['DayofWeek'].map(lambda x: 1 if x == 'Monday' else 2 if x == 'Tuesday' else 3 if x == 'Wednesday' else 4 if x == 'Thursday' else 5 if x == 'Friday' else 6 if x == 'Saturday' else 7 if x == 'Sunday' else np.nan)

#%%

#EDA

# Crimdedf: barchart of frequency by district, barchart of frequency by month, barchart of frequency by hour

#%%

#Countplot for hour
sns.countplot(x ='Hour', data = crimedf,palette = "Set2").set(title='Countplot for frequncy of crime during differet hours')

#%%

#Countplot for month
sns.countplot(x ='Month', data = crimedf,palette = "Set2").set(title='Countplot for frequncy of crime during differet months')

#%%

#Countplot for day of week
sns.countplot(x ='DayofWeek', data = crimedf,palette = "Set2").set(title='Countplot for frequncy of crime during days of week')

#%%

#pie chart for day of week
plt.pie(crimedf['DayofWeek'].value_counts().values, labels=crimedf['DayofWeek'].value_counts().index, colors=sns.color_palette('colorblind'), autopct='%.2f')
plt.title("Pie chart showing crime during different days of the week",bbox={'facecolor':'1.0', 'pad':5})
plt.legend(bbox_to_anchor=(2,0.5), loc="center right")
plt.show()

#Countplot for district
sns.countplot(x ='DISTRICT', data = crimedf,palette = "Set2").set(title='Countplot for frequncy of crime in different districts')

#%%
#Geopandas map

street_map = gpd.read_file('/Users/tylerwallett/Downloads/Police_Districts')

crs = {'init': 'epsg:4326'}

geometry = [Point(xy) for xy in zip( crimedf['X'], crimedf['Y'])]

geo_df = gpd.GeoDataFrame(crimedf,
                 crs=crs,
                 geometry= geometry)

classlocation = Point(-77.046370, 38.899110)

gdf_location = gpd.GeoSeries([classlocation], crs = {'init': 'epsg:4326'})

#%%
#Frequency of crimes in DC 2022 by district
fig, ax = plt.subplots(figsize = (15,15))

street_map.plot(ax =ax, color = 'grey', edgecolor = 'black')

geo_df[geo_df["DISTRICT"] == 1].plot(ax=ax, color = 'red', alpha =0.5)
geo_df[geo_df["DISTRICT"] == 2].plot(ax=ax, color = 'blue', alpha =0.5)
geo_df[geo_df["DISTRICT"] == 3].plot(ax=ax, color = 'yellow', alpha =0.5)
geo_df[geo_df["DISTRICT"] == 4].plot(ax=ax, color = 'orange', alpha =0.5)
geo_df[geo_df["DISTRICT"] == 5].plot(ax=ax, color = 'green', alpha =0.5)
geo_df[geo_df["DISTRICT"] == 6].plot(ax=ax, color = 'purple', alpha =0.5)
geo_df[geo_df["DISTRICT"] == 7].plot(ax=ax, color = 'brown', alpha =0.5)

gdf_location.plot(ax=ax, color='black', marker = '*', markersize = 500)

ax.legend(["DISTRICT 1", "DISTRICT 2", "DISTRICT 3", "DISTRICT 4", "DISTRICT 5", "DISTRICT 6", "DISTRICT 7", "CLASSROOM"])
plt.axis('off')
plt.title('Frequency of crimes in DC 2022 by district')



#%%
fig, ax = plt.subplots(figsize = (15,15))

street_map.plot(ax =ax, color = 'grey', edgecolor = 'black')

geo_df[geo_df["DayofWeek"] == 'Monday'].plot(ax=ax, color = 'red', alpha =0.5)

geo_df[geo_df["DayofWeek"] == 'Tuesday'].plot(ax=ax, color = 'blue', alpha =0.5)

geo_df[geo_df["DayofWeek"] == 'Wednesday'].plot(ax=ax, color = 'yellow', alpha =0.5)

geo_df[geo_df["DayofWeek"] == 'Thursday'].plot(ax=ax, color = 'orange', alpha =0.5)

geo_df[geo_df["DayofWeek"] == 'Friday'].plot(ax=ax, color = 'green', alpha =0.5)

geo_df[geo_df["DayofWeek"] == 'Saturday'].plot(ax=ax, color = 'purple', alpha =0.5)

geo_df[geo_df["DayofWeek"] == 'Sunday'].plot(ax=ax, color = 'brown', alpha =0.5)

gdf_location.plot(ax=ax, color='black', marker = '*', markersize = 500)

plt.axis('off')
plt.title('Frequency of crimes in DC 2022 by district')

#Doesnt really say much...

# %%
