import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
import time

# SUMMARY: Uses geopy to query, extract, and append coordinates of all locations
# in data/cleandata/SolarIncomplete.csv dataset to a new csv.
# Saves the completed file at data/cleandata/SolarComplete.csv

df = pd.read_csv('data/cleandata/SolarIncomplete.csv')
locations = df['Location'].tolist()

# IMPORTANT: You must enter in your computer's user agent to execute the http request
useragent = ''
geolocater = Nominatim(user_agent=useragent)
longs = []
lats  = []


# Extracting coordinates for each location using geopy
# Each loop is delayed 1 second, since OpenStreetMap's terms of service allow max 1 request/sec
for address in locations:
    location = geolocater.geocode(address)
    if location is None:
        raise Exception('Empty lat/long coordinates for: ' + str(address))
    longs.append(location.longitude)
    lats.append(location.latitude)
    time.sleep(1)


# Appending coordinates to df and some further cleanup
df.drop(['Dev1','Dev2'], axis=1, inplace=True)
df['Longitude'] = np.array(longs)
df['Latitude']  = np.array(lats)
df['Power Source'] = 'Solar'
df['Capacity'] = df['Capacity'].str.replace(' MW','', regex=False)
pd.to_numeric(df['Capacity'])
df.to_csv('data/cleandata/SolarComplete.csv', sep= ',', index=False)