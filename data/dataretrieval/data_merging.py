import pandas as pd
import us

# SUMMARY: Cleaning and merging raw Wind data with the SolarComplete.csv after
# we use the Lat/Long to retrieve relevant Wind/Solar Potential from
# govt databases

# WARNING: URLs and filenames will likely change as databases are
# updated, highly advised to use the already prepared clean dataset.

df_wind = pd.read_csv('data/rawdata/WindRawRetrieve.csv', sep=',',low_memory=False)
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

df_solar = pd.read_csv('data/cleandata/SolarComplete.csv')
df_solar.columns = ['State', 'Capacity', 'Longitude', 'Latitude','Power Source']
df_solar['State'] = df_solar['State'].str.extract(state_strings)

state_dict = dict(zip(state_abbrevs,state_names))
df_wind['State'].replace(state_dict,inplace=True)
df_merged = pd.concat([df_solar,df_wind],axis=0,ignore_index=True)
