import shapefile
import numpy as np
import pandas as pd


with shapefile.Reader('data/rawdata/nsrdb_v3_0_1_1998_2016_ghi/nsrdb_v3_0_1_1998_2016_ghi') as shp:
    longitudes = []
    latitudes  = []
    GHI        = []
    for i in range(len(shp)):
        long1 = shp.shape(i).bbox[0]
        long2 = shp.shape(i).bbox[2]
        lat1  = shp.shape(i).bbox[1]
        lat2  = shp.shape(i).bbox[3]
        longitudes += [np.mean([long1,long2])]
        latitudes  += [np.mean([lat1,lat2])]
        GHI        += [shp.record(i)[0]]

with shapefile.Reader('data/rawdata/nsrdb_v3_0_1_1998_2016_dni/nsrdb_v3_0_1_1998_2016_dni') as shp:
    DNI = []
    for i in range(len(shp)):
        DNI += [shp.record(i)[0]]

data = {'Longitude':longitudes, 'Latitude':latitudes, 'DNI':DNI, 'GHI':GHI}
df = pd.DataFrame(data)
df = df[(df.Latitude > 13) & (df.Latitude < 50)]
df = df[(df.Longitude > -172) & (df.Longitude < 145)]
df.to_csv('data/cleandata/SolarPotential.csv',index_label=False)


df_wind = pd.read_csv('data/rawdata/pot_wind_cap_140_current.csv')
df_wind.drop(labels=['FID','gid','area_km2','region'],axis=1,inplace=True)
df_wind.columns = ['Latitude','Longitude','State','a30','a35','a40']
df_wind.to_csv('data/cleandata/WindPotential.csv',index_label=False)

