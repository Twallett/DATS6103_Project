#%%
#Importing modules 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime



#%%
#Importing datasets 

crime = pd.read_csv('Crime_Incidents_in_2022.csv')
stop = pd.read_csv('Stop_Data.csv')


# %%
#Selecting columns of interest 

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




#%%
#EDA

# Crimdedf: barchart of frequency by district, barchart of frequency by month, barchart of frequency by hour



#%%
#Modeling 



#Conclusions 