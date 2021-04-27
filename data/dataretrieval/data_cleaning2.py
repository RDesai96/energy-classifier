import pandas as pd
import us

# SUMMARY: Cleaning and merging raw wind data with the SolarComplete.csv
# Resulting csv saved at data/cleandata/MergedIncomplete.csv

# WARNING: URLs and filenames will likely change as databases are
# updated, highly advised to use the already prepared clean dataset.

raw_wind = 'data/rawdata/WindRawRetrieve1.csv'
full_solar = 'data/cleandata/SolarComplete.csv'
merg_inc = 'data/cleandata/MergedIncomplete.csv'

df_wind = pd.read_csv(raw_wind, sep=',',low_memory=False)
keep_cols = ['t_state','p_name','p_cap','xlong','ylat']
df_wind.drop(df_wind.columns.difference(keep_cols),axis=1,inplace=True)

df_wind = df_wind.sort_values(['t_state','p_name'])\
    .groupby(['t_state','p_name'], sort=False)\
    .aggregate('mean').reset_index()

df_wind.drop('p_name',axis=1,inplace=True)
df_wind.columns = ['State','Capacity','Longitude','Latitude']
df_wind['Power Source'] = 'Wind'

states = us.STATES
state_names = [state.name for state in states]
state_strings = '({})'.format('|'.join(state_names))
state_abbrevs = [state.abbr for state in states]

df_solar = pd.read_csv(full_solar)
df_solar.columns = ['State', 'Capacity', 'Longitude', 'Latitude','Power Source']
df_solar['State'] = df_solar['State'].str.extract(state_strings)

state_dict = dict(zip(state_abbrevs,state_names))
df_wind['State'].replace(state_dict,inplace=True)

df_merged = pd.concat([df_solar,df_wind],axis=0,ignore_index=True)
df_merged.drop(df_merged[df_merged.State == 'Alaska'].index,inplace=True)
df_merged.drop(df_merged[df_merged.State == 'Hawaii'].index,inplace=True)
df_merged.drop(df_merged[df_merged.State == 'GU'].index,inplace=True)
df_merged.drop(df_merged[df_merged.State == 'PR'].index,inplace=True)
df_merged = df_merged.reset_index(drop=True)
df_merged.to_csv(merg_inc,sep=',',index=False)