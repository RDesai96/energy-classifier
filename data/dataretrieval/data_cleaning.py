import csv
import os
import pandas as pd

# SUMMARY:
# Cleans the scraped solar dataset obtained by data_scraping.py
# Saves the cleaned file at data/cleandata/SolarIncomplete.csv

# WARNING: URLs and filenames will likely change as databases are
# updated, highly advised to use the already prepared clean dataset.
# As you see below, lot of hard-coding involved here, not very elegant.
# See README for link to the prepared, cleaned data.

dest_file = 'data/cleandata/SolarIncomplete.csv'
raw_file  = 'data/rawdata/SolarRawRetrieve.csv'
temp_file = 'data/cleandata/SolarCleaning.csv'

if os.path.exists(dest_file):
    os.remove(dest_file)

if os.path.exists(temp_file):
    os.remove(temp_file)

with open(raw_file, 'r') as f:
    for line in f:
        with open(temp_file, 'a') as C:
            C.writelines(line)

# Extracting specific columns from the raw retrieve file
with open(temp_file) as D:
    reader = csv.reader(D, delimiter='\t')
    data_tuples = [(row[0] + ',' + row[1], row[-2], row[-4], row[-3]) for row in reader]



# Getting rid of extraneous strings
# Getting rid of unverifiable projects and adding locations to verified projects

df = pd.DataFrame(data_tuples, columns = ['Location', 'Capacity', 'Dev1', 'Dev2'])

unwanted_strings1 = ['\xa0', ',Operating' , ',Private', ',Under Development', ',Public',
                    ',Under Development','Public (State Owned)', ',Public (City Owned)' ,
                    ',Under Construction', ',Photovoltaic (PV)','(City Owned)','(State Owned)']

pattern1 = '|'.join(unwanted_strings1)
df['Location'] = df['Location'].str.replace(pattern1, '', regex=True)


# Unverifiable/verified projects identified by manual google searches (yup, very painful)
df.iloc[[33,34],0]    = 'New Mexico,Lea County'
df.iloc[[35,67,192],0]    = 'New Mexico,Eddy County'
df.iloc[66,1]  = '110 MW'
df.iloc[[116,210,211,212],0]   = 'Mountain Home Air Force Base,Idaho'
df.iloc[118,0] = 'California,Mojave Desert'
df.iloc[154,0] = 'Abilene,Texas'
df.iloc[[216,219],0] = 'California,San Diego County'
df.iloc[[221,222,223],0] = 'Indianapolis,Indiana'
df.iloc[227,0] = 'Webberville,Texas'
df.iloc[[238,239,240],0]  = 'Arizona,Maricopa County'
df.iloc[243,0] = 'San Luis,Arizona'
df.iloc[264,0] = 'Delaware,Kent County'
df.iloc[265,0] = 'Riverside,California'
df.iloc[312,0] = 'Gila Bend,Arizona'
df.iloc[370,0] = 'El Paso,Texas'
df.iloc[396,0] = 'Santa Margarita,California'
df.iloc[[339,340,341,399],0] = 'California,Fresno County'
df.iloc[400,0] = 'California,Madera County'
df.iloc[401,0] = 'California,Imperial County'
df.iloc[402,0] = 'Boulder City,Nevada'
df.iloc[403,0] = 'Panoche Valley,California'
df.iloc[403,1] = '130 MW'
df.iloc[404,0] = 'California,Imperial County'
df.iloc[405,0] = 'Alpaugh,California'
df.iloc[408,0] = 'Fort Mohave,Arizona'
df.drop(df.index[[108,109,111,131,155,189,261,278,362,369,372,394,397,398]], inplace=True)


df.to_csv(dest_file, sep = ',',index=False)