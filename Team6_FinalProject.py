#%%
#Importing modules 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



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




#%%
#EDA

# Crimdedf: barchart of frequency by district, barchart of frequency by month, barchart of frequency by hour



#%%
#Modeling 



#Conclusions 